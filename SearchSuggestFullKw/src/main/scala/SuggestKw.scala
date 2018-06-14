/**
  *
  * author: jomei
  * date: 2018/6/5 16:33
  */
//1)import and setting
import scala.math.pow
import scala.math.abs
import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import java.util.regex.Matcher
import java.util.regex.Pattern
case class TempRow(label: Int, dt: String)

object SuggestKw extends Serializable{
  def main(args: Array[String]):Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Generate hot for kw")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)initial variable
    val date_end = args(0) //first argument is yesterday
    val period = 14 // for burst and hot
    val moving_average_length = 7
    val hot_length = period - 2
    //filter_tag starts from 1, so hot_length will ensure the recent three days

    val date_period = getDaysPeriod(date_end, period - 1)
    //create df_date to fill the missing dt
    var date_list_buffer = new ListBuffer[TempRow]()
    for (dt <- date_period){
      date_list_buffer += TempRow(1, dt)
    }
    val date_list = date_list_buffer.toList
    val df_date = date_list.toDF
    df_date.createOrReplaceTempView("date_table")

    //3)read date and create df_period with kw,hot,dt,label without missing dt
    val sql_kw_period = s"select dt, kw, hot from temp.search_fourteen_day_full_kw where cdt='$date_end'"
    val df_kw_period = spark.sql(sql_kw_period)
    df_kw_period.persist()

    val df_all = df_kw_period.withColumn("label", lit(1))
                             .filter($"kw".isNotNull && $"kw" =!= "" && $"kw" =!= " ")

    df_all.createOrReplaceTempView("table_all")
    val sql_period  = "select kw, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
    val df_period_temp = spark.sql(sql_period)
    // to eliminate the extra data because of RIGHT OUTER JOIN
    val df_period = df_period_temp.groupBy("dt","kw").agg(max("hot").as("hot"))
    df_period.persist()

    //4)calculate the burst score and hot score, create df_cal_save with kw, burst, hot
    //use window function
    val weights = getWeight(moving_average_length)
    val index = List.range(0,moving_average_length)
    val window_ma = Window.partitionBy("kw").orderBy(asc("dt"))
    val period_ma = df_period.withColumn("weightedmovingAvg", weighted_average(index, weights, window_ma, df_period("hot")))
    //filter the last n rows, cause their weighted moving average we didn't calculate, cause data is missing
    val window_filter = Window.partitionBy("kw").orderBy(asc("dt"))
    val period_filter_temp = period_ma.withColumn("filter_tag", row_number.over(window_filter))
    period_filter_temp.persist()
    val period_filter = period_filter_temp.filter($"filter_tag" >= moving_average_length)
    //calculate the amp score
    val period_burst = period_filter.groupBy("kw").agg((last("weightedmovingAvg")-(mean("weightedmovingAvg") + lit(1.5)*stddev_samp("weightedmovingAvg"))).as("weightedamp"))

    //calculate the recent three days total hot
    //filter_tag starts from 1, so hot_length will ensure the recent three days
    val period_filter_hot = period_filter_temp.filter($"filter_tag" >= hot_length)
    val period_hot = period_filter_hot.groupBy("kw").agg(sum("hot") as "hot")

    //combine both(weightedamp and hot) of them together
    val period_cal = period_burst.as("d1")
                                 .join(period_hot.as("d2"), $"d1.kw" === $"d2.kw")
                                 .select($"d1.*",$"d2.hot")

    val df_cal_save = period_cal.withColumn("burst", bround($"weightedamp", 3))
                                .withColumn("hotTmp", $"hot".cast(IntegerType))
                                .drop("hot")
                                .withColumnRenamed("hotTmp", "hot")
                                .select("kw", "burst", "hot")
    df_cal_save.persist()
    //unpersist to release memory
    df_period.unpersist()
    df_kw_period.unpersist()
    period_filter_temp.unpersist()

    //6)save burst and its songname
    //it will be used to check
    df_cal_save.createOrReplaceTempView("table_burst_name")
    val df_burst_name_save_create = """
create table if not exists temp.jimmy_dt_burst_name_kw
(
    kw string,
    burst DOUBLE,
    hot INT
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(df_burst_name_save_create)

    val df_burst_name_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_burst_name_kw PARTITION(cdt='$date_end') select kw, burst, hot from table_burst_name
"""
    spark.sql(df_burst_name_save)

    //7)standard the data, (data - mean)/std
    val df_variable = df_cal_save.select(mean(df_cal_save("hot")),stddev_samp(df_cal_save("hot")),mean(df_cal_save("burst")),stddev_samp(df_cal_save("burst")))
    val (hot_mean, hot_std, burst_mean, burst_std) = (df_variable.first.getDouble(0), df_variable.first.getDouble(1), df_variable.first.getDouble(2), df_variable.first.getDouble(3))
    //it will run at once
    val df_cal = df_cal_save.withColumn("burst_standard", ($"burst"-burst_mean)/burst_std)
                            .withColumn("hot_standard", ($"hot"-hot_mean)/hot_std)
    df_cal.persist()
    //calculate the dynamic value(hot_coefficient) to combine burst and hot
    val coefficient_temp = math.abs(df_cal.sort($"hot_standard".desc).limit(20).agg(mean($"hot_standard")).first.getDouble(0) /(df_cal.sort($"burst_standard".desc).limit(20).agg(mean($"burst_standard")).first.getDouble(0)))
    val coefficient_burst = 1
    var coefficient_temp2 = 0.0
    if (coefficient_temp * coefficient_burst < 1){
      coefficient_temp2 = 1.0
    }
    else if(coefficient_temp * coefficient_burst > 20){
      coefficient_temp2 = 20.0
    }
    else{
      coefficient_temp2 = coefficient_temp * coefficient_burst
    }
    val hot_coefficient = coefficient_temp2

    //to move up the minmum of hot-standard to ensure the sum operation below won't trouble in summing negative value
    val temp_min = df_cal.select(min($"hot_standard"))
                         .first
                         .getDouble(0)
    var hot_min = 0.0
    if( temp_min < 0){
      hot_min = math.abs(temp_min)
    }
    //we only use the positive part of burst_standard
    val df_temp = df_cal.withColumn("positive", when($"burst_standard" > lit(0), $"burst_standard").otherwise(lit(0)))
                        .withColumn("positive2", $"hot_standard" + lit(hot_min))
    val df_kw = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive" + $"positive2").otherwise($"positive" + $"positive2"))
                       .select($"kw", bround($"result_temp", 3) as "hot")
    df_kw.persist()
    df_kw.createOrReplaceTempView("table_kw")

    //8)save values:hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min
    var save_value_buf = new ListBuffer[(Double, Double, Double, Double, Double, Double)]()
    save_value_buf += ((hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min))
    val df_save_value = save_value_buf.toList
                                      .toDF("hot_mean", "hot_std", "burst_mean", "burst_std", "hot_coefficient", "hot_min")

    df_save_value.createOrReplaceTempView("savevalue_table")
    val sql_save_value_create = """
create table if not exists temp.jimmy_dt_save_value_kw
(
    hot_mean DOUBLE,
    hot_std DOUBLE,
    burst_mean DOUBLE,
    burst_std DOUBLE,
    hot_coefficient DOUBLE,
    hot_min DOUBLE
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_save_value_create)

    val sql_save_value = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_save_value_kw PARTITION(cdt='$date_end') select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from savevalue_table
"""
    spark.sql(sql_save_value)


    //9)save result
    val sql_kw_save_create = """
create table if not exists temp.jimmy_dt_hot_score_kw
(
    kw string,
    hot DOUBLE
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_kw_save_create)

    val sql_kw_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_hot_score_kw PARTITION(cdt='$date_end') select kw, hot from table_kw
"""
    spark.sql(sql_kw_save)
    //unpersist to gain memory
    df_cal.unpersist()

    //10)save result with num
    val sql_search_num = """
select
kw,
hot,
resultnum
from table_kw a
left join
(
select keyword, resultnum
from
(
select keyword, resultnum, row_number() over(partition by keyword order by ren desc) as rank
from
(
select kw as keyword, ivar5 resultnum, count(distinct mid) as ren
from
ddl.dt_list_ard_d
where dt="""+s"""'$date_end'"""+"""
and action='search'
and b like '搜索页-%'
and ivar5 is not null
and ivar5 <>'null'
and ivar5 <>''
group by kw, ivar5
)b
)b where rank=1
)b on a.kw = b.keyword
"""
    val df_search_num = spark.sql(sql_search_num)
    df_search_num.createOrReplaceTempView("num_search_savetable")
    val sql_result_num_create= """
create table if not exists temp.jimmy_dt_hot_result_num_mix_kw
(
    kw string,
    hot DOUBLE,
    resultnum INT
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_result_num_create)

    val sql_result_num= s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_num_mix_kw PARTITION(cdt='$date_end') select kw, hot, resultnum from num_search_savetable
"""
    spark.sql(sql_result_num)
    spark.stop() //to avoid ERROR LiveListenerBus: SparkListenerBus has already stopped! Dropping event SparkListenerExecutorMetricsUpdate
  }

  def getDaysPeriod(dt: String, interval: Int): List[String] = {
    var period = new ListBuffer[String]() //initialize the return List period
    period += dt
    val cal: Calendar = Calendar.getInstance() //reset the date in Calendar
    cal.set(dt.split("-")(0).toInt, dt.split("-")(1).toInt - 1, dt.split("-")(2).toInt)
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd") //format the output date
    for (i <- 0 to interval - 1){
      cal.add(Calendar.DATE, - 1)
      period += dateFormat.format(cal.getTime())
    }
    period.toList
  }
  def getWeight(length: Int): List[Double]= {
    var sum = 0.0
    for (i <- 0 to length-1 ){
      sum += pow(0.5, i)
    }
    //    val weights = for (i <- 0 to length-1 ) yield pow(0.5, i)/sum // it will return scala.collection.immutable.IndexedSeq
    val weights = for (i <- List.range(0, length) ) yield pow(0.5, i)/sum
    //    var weights_buffer = new ListBuffer[Double]()
    //    for (i <- 0 to length-1 ){
    //      weights_buffer += pow(0.5, i)/sum
    //    }
    //    val weights = weights_buffer.toList
    weights
  }
  def weighted_average(index: List[Int], weights: List[Double], w: WindowSpec, c: Column): Column= {
    val wma_list = for (i <- index) yield (lag(c, i).over(w))*weights(i) // list comprehension, map also can do some easy thing, return scala.collection.immutable.IndexedSeq
    wma_list.reduceLeft(_ + _)
  }
}
