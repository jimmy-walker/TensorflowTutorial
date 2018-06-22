/**
  *
  * author: jomei
  * date: 2018/6/22 16:00
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
object SuggestKwExp extends Serializable{
  def main(args: Array[String]): Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Generate searchsuggest for recent three days kw")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)initial variable
    val date_end = args(0) //first argument is yesterday

    val date_period = getDaysPeriod(date_end, 2) //take three days record
    val date_start = date_period.takeRight(1)(0)
    var date_list_buffer = new ListBuffer[TempRow]()
    for (dt <- date_period){
      date_list_buffer += TempRow(1, dt)
    }
    val date_list = date_list_buffer.toList
    val df_date = date_list.toDF
    df_date.createOrReplaceTempView("date_table")


    //3)raw data sql and dataframe save
    val sql_ard_scid=s"select dt, kw, hot from temp.search_three_day_full_kw where cdt='$date_end'"
    val df_ard_scid_sep = spark.sql(sql_ard_scid)
    val df_ard_scid = df_ard_scid_sep.groupBy("dt","kw").agg(sum("hot").as("hot"))
//    df_ard_scid.persist()

    //4)merge all and groupby to df_all with kw,hot,dt,label
    //then fulfill the missing dt
    //from df_all to generate df_period with complete dt
    val df_all = df_ard_scid.withColumn("label", lit(1))

    df_all.createOrReplaceTempView("table_all")

    val sql_period  = "select kw, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
    val df_period_temp = spark.sql(sql_period)
    // to eliminate the extra data because of RIGHT OUTER JOIN
    val df_period = df_period_temp.groupBy("dt","kw").agg(max("hot").as("hot"))
    df_period.persist()

    //5)calculate the final score

    //use window function
//    val weights = List(1, 0.1, 0.01)
    val weights = List(args(1).toDouble, args(2).toDouble, args(3).toDouble)
    val index = List.range(0,3)
    val window_sum = Window.partitionBy("kw").orderBy(asc("dt"))
    val period_sum = df_period.withColumn("DecayedSum", weighted_average(index, weights, window_sum, df_period("hot")))
    //filter the last n rows, cause their weighted moving average we didn't calculate, cause data is missing
    val window_filter = Window.partitionBy("kw").orderBy(asc("dt"))
    val period_filter_temp = period_sum.withColumn("filter_tag", row_number.over(window_filter))
//    period_filter_temp.persist()
    val period_filter = period_filter_temp.filter($"filter_tag" >= 3)
                                          .withColumn("hot_new", bround($"DecayedSum", 2))
    period_filter.persist()
    period_filter.createOrReplaceTempView("table_kw")

    //6)save result
    val sql_kw_save_create = """
create table if not exists temp.jimmy_dt_three_hot_score_kw
(
    kw string,
    hot DOUBLE
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_kw_save_create)

    val sql_kw_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_three_hot_score_kw PARTITION(cdt='$date_end') select kw, hot_new from table_kw
"""
    spark.sql(sql_kw_save)

    //7)save result with num
    val sql_search_num = """
select
kw,
hot_new,
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
create table if not exists temp.jimmy_dt_three_hot_result_num_mix_kw
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
INSERT OVERWRITE TABLE temp.jimmy_dt_three_hot_result_num_mix_kw PARTITION(cdt='$date_end') select kw, hot_new, resultnum from num_search_savetable
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
