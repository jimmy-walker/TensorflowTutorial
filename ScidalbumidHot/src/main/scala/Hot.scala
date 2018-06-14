/**
  *
  * author: jomei
  * date: 2018/5/29 15:41
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
object Hot extends Serializable{

  def main(args: Array[String]):Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Generate result for scid_albumid")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)initial variable
    val date_end = args(0) //first argument is yesterday
    val penality = 0.1
    val period = 14 // for burst and hot
    val moving_average_length = 7
    val hot_length = period - 2
    //filter_tag starts from 1, so hot_length will ensure the recent three days
    val category = List("ost", "插曲", "主题曲", "原声带", "配乐", "片尾曲",
      "片头曲", "originalsoundtrack", "原声", "宣传曲", "op",
      "ed", "推广曲", "角色歌", "in", "背景音乐", "tm", "钢琴曲",
      "开场曲", "剧中曲", "bgm", "暖水曲", "主题歌")
    val broadcasted = spark.sparkContext.broadcast(category)
    val alias = Map("^G.E.M.邓紫棋$|^G.E.M.邓紫棋(?=、)|(?<=、)G.E.M.邓紫棋(?=、)|(?<=、)G.E.M.邓紫棋$" -> "邓紫棋")
    val broadcasted_alias = spark.sparkContext.broadcast(alias)

    val date_period = getDaysPeriod(date_end, period - 1)
    //create df_date to fill the missing dt
    var date_list_buffer = new ListBuffer[TempRow]()
    for (dt <- date_period){
      date_list_buffer += TempRow(1, dt)
    }
    val date_list = date_list_buffer.toList
    val df_date = date_list.toDF
    df_date.createOrReplaceTempView("date_table")

    //3)function
    //define strip function
    val mystrip = udf{(a:String) =>
      if(a != null & a != ""){
        a.toLowerCase.replaceAll( "[\\p{P}+~$`^=|<>～｀＄＾＋＝｜＜＞￥×〾〿 ]" , "")
      }
      else {""}
    }
    //define alias function
    val myalias = udf{(a:String) =>
      if(a != null & a != ""){
        var result = ""
        var str = a.trim
        for ((k,v) <- broadcasted_alias.value){
          val pattern = Pattern.compile(k)
          val matcher = pattern.matcher(str)
          if (matcher.find()){
            str = str.substring(0, matcher.start()) + v + str.substring(matcher.end(), str.length)
            result = str
          }
        }
        result
      }
      else {""}
    }
    //define category function
    val myfunc = udf{(a:String) =>
      if(a != null & a != ""){
        var end = -1
        var result = ""
        for (term <- broadcasted.value){
          val position = a.toLowerCase.replace(".", "").replace(" ", "").indexOf(term)
          if (position > -1){
            if (position + term.length > end){
              result = term
              end = position + term.length
            }
            if (position + term.length == end){
              if (term.length > result.length){
                result = term
                end = position + term.length
              }
            }
          }
        }
        result
      }
      else {""}
    }

    //4)read date and create df_period with scid_albumid,hot,dt,label without missing dt
    val sql_scid_period = s"select dt, scid_albumid, hot from temp.search_fourteen_day_full where cdt='$date_end'"
    val df_scid_period = spark.sql(sql_scid_period)
    df_scid_period.persist()

    val df_all = df_scid_period.withColumn("label", lit(1))
    df_all.createOrReplaceTempView("table_all")
    val sql_period  = "select scid_albumid, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
    val df_period_temp = spark.sql(sql_period)
    // to eliminate the extra data because of RIGHT OUTER JOIN
    val df_period = df_period_temp.groupBy("dt","scid_albumid").agg(max("hot").as("hot"))
    df_period.persist()

    //5)calculate the burst score and hot score, create df_cal_save with scid_albumid, burst, hot
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
    //unpersist to release memory
    df_period.unpersist()
    df_scid_period.unpersist()
    period_filter_temp.unpersist()

    //6)standard the data, (data - mean)/std
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
    val df_scid = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive" + $"positive2").otherwise($"positive" + $"positive2"))
                         .select($"scid_albumid", bround($"result_temp", 3) as "hot")

    //save values:hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min
    var save_value_buf = new ListBuffer[(Double, Double, Double, Double, Double, Double)]()
    save_value_buf += ((hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min))
    val df_save_value = save_value_buf.toList
                                      .toDF("hot_mean", "hot_std", "burst_mean", "burst_std", "hot_coefficient", "hot_min")

    df_save_value.createOrReplaceTempView("savevalue_table")
    val sql_save_value_create = """
create table if not exists temp.jimmy_dt_save_value
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
INSERT OVERWRITE TABLE temp.jimmy_dt_save_value PARTITION(cdt='$date_end') select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from savevalue_table
"""
    spark.sql(sql_save_value)

    df_scid.persist()
    df_scid.createOrReplaceTempView("table_scid")
    //it will be used later
    val sql_scid_save_create = """
create table if not exists temp.jimmy_dt_hot_score
(
        scid_albumid string,
        hot DOUBLE
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_scid_save_create)

    val sql_scid_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_hot_score PARTITION(cdt='$date_end') select scid_albumid, hot from table_scid
"""
    spark.sql(sql_scid_save)
    //unpersist to gain memory
    df_cal.unpersist()


    //7)using df_scid to extract name from common.st_k_mixsong_part
    //select singername, songname, albumname
    val sql_sn=s"""
select a.mixsongid, a.choric_singer, a.songname, a.albumname, b.hot
from common.st_k_mixsong_part a
inner join
table_scid b
where a.dt = '$date_end' and a.mixsongid=b.scid_albumid
"""
    val df_sn = spark.sql(sql_sn)
    //filter and format name
    val df_sn_sep = df_sn.withColumn("song", regexp_replace($"songname", "[ ]*\\([^\\(\\)]*\\)$", ""))
                         .withColumn("kind", regexp_extract($"songname", "\\(([^\\(\\)]*)\\)$", 1))
                         .withColumn("singer", regexp_replace($"choric_singer", "[ ]*\\([0-9]*\\)$", "")) //to eliminate the duplicate singer effect
//                         .withColumnRenamed("choric_singer", "singer") //to eliminate the space effect

    df_sn_sep.persist()

    df_sn_sep.createOrReplaceTempView("table_sep")
    //it will be used later
    val sql_sn_sep_save_create = """
create table if not exists temp.jimmy_dt_sn_sep
(
        mixsongid string,
        albumname string,
        song string,
        kind string,
        singer string,
        hot DOUBLE
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_sn_sep_save_create)

    val sql_sn_sep_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_sn_sep PARTITION(cdt='$date_end') select mixsongid, albumname, song, kind, singer, hot from table_sep
"""
    //remember that we should select the item in the hive item order!!!!!!!!!!!!!!!
    spark.sql(sql_sn_sep_save)

    //8)save burst and its songname
    val df_burst_name = df_cal_save.as("d1")
                                   .join(df_sn_sep.as("d2"), $"d1.scid_albumid" === $"d2.mixsongid", "left")
                                   .select($"d1.*", $"d2.song", $"d2.singer", $"d2.kind", $"d2.albumname")

    //it will be used to check
    df_burst_name.createOrReplaceTempView("table_burst_name")
    val df_burst_name_save_create = """
create table if not exists temp.jimmy_dt_burst_name
(
        scid_albumid string,
        burst DOUBLE,
        hot INT,
        song string,
        singer string,
        kind string,
        albumname string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(df_burst_name_save_create)

    val df_burst_name_save = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_burst_name PARTITION(cdt='$date_end') select scid_albumid, burst, hot, song, singer, kind, albumname from table_burst_name
"""
    spark.sql(df_burst_name_save)

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
