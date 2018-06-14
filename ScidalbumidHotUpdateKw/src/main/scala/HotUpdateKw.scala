/**
  *
  * author: jomei
  * date: 2018/6/7 20:22
  */
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
object HotUpdateKw extends Serializable{
  def main(args: Array[String]): Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Generate hot for scid_albumid update")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)initial variable
    val date_today = args(0) //first argument is yesterday
    val update_time = args(1) //second argument is update_time
    val penality = 0.1
    val penality2 = 2 //for kw, we change it, cause scid_albumid stands for multiple kw
    val period = 14 // for burst and hot
    val moving_average_length = 7
    val hot_length = period - 2
    //filter_tag starts from 1, so hot_length will ensure the recent three days

    val date_period = getDaysPeriod(date_today, period - 1)
    val date_start = date_period.takeRight(1)(0)
    val date_period_ystd = getDaysPeriod(date_today, 1) //take yesterday
    val date_end = date_period_ystd.takeRight(1)(0)
    var date_list_buffer = new ListBuffer[TempRow]()
    for (dt <- date_period){
      date_list_buffer += TempRow(1, dt)
    }
    val date_list = date_list_buffer.toList
    val df_date = date_list.toDF
    df_date.createOrReplaceTempView("date_table")

    //3)read date and create df_period with scid_albumid,hot,dt,label without missing dt
    val sql_scid_period = s"select dt, scid_albumid, hot from temp.search_fourteen_day_update_kw_scid where cdt='$date_today' and time = '$update_time'"
    val df_scid_period = spark.sql(sql_scid_period)
    df_scid_period.persist()

    val df_all = df_scid_period.withColumn("label", lit(1))

    df_all.createOrReplaceTempView("table_all")
    val sql_period  = "select scid_albumid, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
    val df_period_temp = spark.sql(sql_period)
    // to eliminate the extra data because of RIGHT OUTER JOIN
    val df_period = df_period_temp.groupBy("dt","scid_albumid").agg(max("hot").as("hot"))
    df_period.persist()

    //4)calculate the burst score and hot score, create df_cal_save with scid_albumid, burst, hot
    //use window function
    val weights = getWeight(moving_average_length)
    val index = List.range(0,moving_average_length)
    val window_ma = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
    val period_ma = df_period.withColumn("weightedmovingAvg", weighted_average(index, weights, window_ma, df_period("hot")))
    //filter the last n rows, cause their weighted moving average we didn't calculate, cause data is missing
    val window_filter = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
    val period_filter_temp = period_ma.withColumn("filter_tag", row_number.over(window_filter))
    period_filter_temp.persist()
    val period_filter = period_filter_temp.filter($"filter_tag" >= moving_average_length)
    //calculate the amp score
    val period_burst = period_filter.groupBy("scid_albumid").agg((last("weightedmovingAvg")-(mean("weightedmovingAvg") + lit(1.5)*stddev_samp("weightedmovingAvg"))).as("weightedamp"))

    //calculate the recent three days total hot
    //filter_tag starts from 1, so hot_length will ensure the recent three days
    val period_filter_hot = period_filter_temp.filter($"filter_tag" >= hot_length)
    val period_hot = period_filter_hot.groupBy("scid_albumid").agg(sum("hot") as "hot")

    //combine both(weightedamp and hot) of them together
    val period_cal = period_burst.as("d1")
      .join(period_hot.as("d2"), $"d1.scid_albumid" === $"d2.scid_albumid")
      .select($"d1.*",$"d2.hot")

    val df_cal_save = period_cal.withColumn("burst", bround($"weightedamp", 3))
      .withColumn("hotTmp", $"hot".cast(IntegerType))
      .drop("hot")
      .withColumnRenamed("hotTmp", "hot")
      .select("scid_albumid", "burst", "hot")
    df_cal_save.persist()

    //it will be used later
    val sql_burst_hot_save_create = """
create table if not exists temp.jimmy_dt_burst_hot_score_update_kw_scid
(
scid_albumid string,
burst DOUBLE,
hot INT
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_burst_hot_save_create)

    df_cal_save.createOrReplaceTempView("cal_savetable")
    val sql_burst_hot_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_burst_hot_score_update_kw_scid PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, burst, hot from cal_savetable"
    spark.sql(sql_burst_hot_save)

    //5)standard
    //use save value from yesterday
    val sql_read_value = s"select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from temp.jimmy_dt_save_value_kw where cdt='$date_end'"
    val df_save_value = spark.sql(sql_read_value)
    val (hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min) = (df_save_value.first.getDouble(0), df_save_value.first.getDouble(1), df_save_value.first.getDouble(2), df_save_value.first.getDouble(3), df_save_value.first.getDouble(4), df_save_value.first.getDouble(5))

    //standard the data, (data - mean)/std
    //we only use the burst score and one day hot score, cause we ignore the hot score it's meaningless to calculate the recent three day in update operation
    val df_cal = df_cal_save.filter($"burst" > 0) //only extract the positive one
      .withColumn("burst_standard", when($"burst" > lit(50), (($"burst"*24.0*penality2/(update_time.toInt))-burst_mean)/burst_std).otherwise(($"burst"-burst_mean)/burst_std))
      .withColumn("hot_standard", ($"hot"*24.0/(update_time.toInt)-hot_mean)/hot_std)

    //we filter it above zero
    //save both burst and hot
    val df_temp = df_cal.withColumn("positive", when($"burst_standard" > lit(0), $"burst_standard").otherwise(lit(0)))
      .withColumn("positive2", $"hot_standard" + lit(hot_min))
      .filter($"positive" > 0 && $"positive2" > 0) //we only calculate the bursting song
    // we consier not to add lit(1) as bias
    val df_scid = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive"  + $"positive2").otherwise($"positive" + $"positive2"))
      .select($"scid_albumid", bround($"result_temp", 3) as "hot")

    df_scid.persist()
    df_scid.createOrReplaceTempView("table_scid_update")
    val sql_scid_save_create = """
create table if not exists temp.jimmy_dt_hot_score_kw_update_scid
(
    scid_albumid string,
    hot DOUBLE
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_scid_save_create)

    val sql_scid_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_hot_score_kw_update_scid PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, hot from table_scid_update
"""
    spark.sql(sql_scid_save)

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
