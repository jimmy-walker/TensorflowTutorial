/**
  *
  * author: jomei
  * date: 2018/6/22 16:52
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
class HotUpdateKwExp extends Serializable{
  def main(args: Array[String]): Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Calculate hot for updated scid_albumid")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)variable
    val date_today = args(0)
    val update_time = args(1)
    val compensate = 1
    val date_period = getDaysPeriod(date_today, 2) //take three days record
    val date_start = date_period.takeRight(1)(0)

    var date_list_buffer = new ListBuffer[TempRow]()
    for (dt <- date_period){
      date_list_buffer += TempRow(1, dt)
    }
    val date_list = date_list_buffer.toList
    val df_date = date_list.toDF
    df_date.createOrReplaceTempView("date_table")

    //3)raw data sql and dataframe save
    val sql_scid_period = s"select dt, scid_albumid, hot from temp.search_threeexp_day_update_kw_scid where cdt='$date_today' and time = '$update_time'"
    val df_scid_period = spark.sql(sql_scid_period)
    df_scid_period.persist()

    //4)merge all and groupby to df_all with dt,scid_albumid, hot, label
    //then fulfill the missing dt
    //from df_all to generate df_period with complete dt
    val df_all = df_scid_period.withColumn("label", lit(1))

    df_all.createOrReplaceTempView("table_all")

    val sql_period  = "select scid_albumid, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
    val df_period_temp = spark.sql(sql_period)
    // to eliminate the extra data because of RIGHT OUTER JOIN
    val df_period = df_period_temp.groupBy("dt","scid_albumid").agg(max("hot").as("hot"))
    df_period.persist()

    //5)calculate the final score
    //use window function
    // val weights = List(1.0, 0, 0)
    val weights_comp = List(1*24.0*compensate/(update_time.toInt), 0, 0)
    // val weights_comp2 = List(1*24.0*2/(update_time.toInt), 0, 0)
    // val weights_comp3 = List(1*24.0*3/(update_time.toInt), 0, 0)
    // val weights_comp4 = List(1*24.0*4/(update_time.toInt), 0, 0)

    val index = List.range(0,3)
    val window_sum = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
    val period_sum = df_period.withColumn("DecayedSumComp", weighted_average(index, weights_comp, window_sum, df_period("hot")))
    // .withColumn("DecayedSum", weighted_average(index, weights, window_sum, df_period("hot")))
    // .withColumn("DecayedSumComp2", weighted_average(index, weights_comp2, window_sum, df_period("hot")))
    // .withColumn("DecayedSumComp3", weighted_average(index, weights_comp3, window_sum, df_period("hot")))
    // .withColumn("DecayedSumComp4", weighted_average(index, weights_comp4, window_sum, df_period("hot")))

    //filter the last n rows, cause their weighted moving average we didn't calculate, cause data is missing
    val window_filter = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
    val period_filter_temp = period_sum.withColumn("filter_tag", row_number.over(window_filter))
    period_filter_temp.persist()
    val period_filter = period_filter_temp.filter($"filter_tag" >= 3)
                                          .filter($"DecayedSumComp" > 0)
    period_filter.persist()
    period_filter.createOrReplaceTempView("table_scid_update")

    //6)save result
    val sql_scid_save_create = """
create table if not exists temp.jimmy_dt_three_hot_score_kw_update_scid
(
    scid_albumid string,
    hot DOUBLE
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_scid_save_create)

    val sql_scid_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_three_hot_score_kw_update_scid PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, DecayedSumComp from table_scid_update
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
