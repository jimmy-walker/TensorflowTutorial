/**
  *
  * author: jomei
  * date: 2018/6/1 17:00
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
object SuggestUpdate extends Serializable{
  def main(args: Array[String]): Unit = {
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

    //2)variable
    //list variable
    val date_today = args(0) //first argument is today
    val update_time = args(1) //second argument is update_time
    val penality = 0.1
    val penality2 = 3
    // val period = 3 // for only hot
    val period = 14 // for burst and hot
    val moving_average_length = 7
    val hot_length = period - 1 //only calculate one day, ie two days totally
    //filter_tag starts from 1, so hot_length will ensure the recent three days
    val category = List("ost", "插曲", "主题曲", "原声带", "配乐", "片尾曲",
      "片头曲", "originalsoundtrack", "原声", "宣传曲", "op",
      "ed", "推广曲", "角色歌", "in", "背景音乐", "tm", "钢琴曲",
      "开场曲", "剧中曲", "bgm", "暖水曲", "主题歌")
    val broadcasted = spark.sparkContext.broadcast(category)
    val alias = Map("^G.E.M.邓紫棋$|^G.E.M.邓紫棋(?=、)|(?<=、)G.E.M.邓紫棋(?=、)|(?<=、)G.E.M.邓紫棋$" -> "邓紫棋")
    val broadcasted_alias = spark.sparkContext.broadcast(alias)
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

    //4)read new song info(df_sn_sep); new song burst(df_cal_save)
    val sql_sn = s"select mixsongid, choric_singer, songname, albumname, hot from temp.jimmy_dt_sn_update where cdt='$date_today'"
    val df_sn = spark.sql(sql_sn)
    //filter and format name
    val df_sn_sep = df_sn.withColumn("song", regexp_replace($"songname", "[ ]*\\([^\\(\\)]*\\)$", ""))
                         .withColumn("kind", regexp_extract($"songname", "\\(([^\\(\\)]*)\\)$", 1))
                         .withColumnRenamed("choric_singer", "singer") //to eliminate the space effect
    df_sn_sep.persist()

    val sql_burst_hot_read = s"select scid_albumid, burst, hot from temp.jimmy_dt_burst_hot_score_update where cdt='$date_today' and time='$update_time'"
    val df_cal_save = spark.sql(sql_burst_hot_read)

    //save burst and its songname
    val df_burst_name = df_cal_save.as("d1")
                                   .join(df_sn_sep.as("d2"), $"d1.scid_albumid" === $"d2.mixsongid", "left")
                                   .select($"d1.*", $"d2.song", $"d2.singer", $"d2.kind", $"d2.albumname")
                                   .filter($"song".isNotNull) //cause df_scid filter positive burst, it will make some mixsongid of df_sn_sep does not exist

    df_burst_name.createOrReplaceTempView("table_burst_name")
    val df_burst_name_save_create ="""
create table if not exists temp.jimmy_dt_burst_name_update
(
scid_albumid string,
burst DOUBLE,
hot INT,
song string,
singer string,
kind string,
albumname string
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(df_burst_name_save_create)
    val df_burst_name_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_burst_name_update PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, burst, hot, song, singer, kind, albumname from table_burst_name"
    spark.sql(df_burst_name_save)

    //5) generate index
    val df_singersong_alias = df_sn_sep.filter($"singer".isNotNull && $"singer" =!= "" && $"song".isNotNull && $"song" =!= "")
                                       .groupBy("singer", "song").agg(sum("hot")*penality as "result")
                                       .withColumn("term",concat($"singer", lit(" "), $"song"))
                                       .withColumn("alias_singer", myalias($"singer"))
                                       .withColumn("alias", when($"alias_singer" =!= "", concat($"alias_singer", lit(" "), $"song")).otherwise(""))

    val df_singersong = df_singersong_alias.select("term", "result", "alias")

    val df_songsinger_alias = df_sn_sep.filter($"singer".isNotNull && $"singer" =!= "" && $"song".isNotNull && $"song" =!= "")
                                       .groupBy("singer", "song").agg(sum("hot")*penality as "result")
                                       .withColumn("term",concat($"song", lit(" "), $"singer"))
                                       .withColumn("alias_singer", myalias($"singer"))
                                       .withColumn("alias", when($"alias_singer" =!= "", concat($"song", lit(" "), $"alias_singer")).otherwise(""))

    val df_songsinger = df_songsinger_alias.select("term", "result", "alias")

    val df_singer_single_pre = df_sn_sep.filter($"singer".isNotNull && $"singer" =!= "")
                                        .filter(not($"singer".contains("、")))
                                        .groupBy("singer").agg(sum("hot") as "result")
                                        .withColumnRenamed("singer", "term")

    val df_singer_multiple_pre = df_sn_sep.filter($"singer".isNotNull && $"singer" =!= "")
                                          .filter($"singer".contains("、"))
                                          .groupBy("singer").agg(sum("hot") as "pre_result")
                                          .withColumnRenamed("singer", "term")

    val df_singer_multiple = df_singer_multiple_pre.withColumn("result", $"pre_result"*penality*penality) //to avoid same with multiple singer and singer/song
                                                   .select("term", "result")

    val df_singer_multiple_sep = df_singer_multiple_pre.withColumn("singer_sep", explode(split($"term", "、")))
                                                       .select("pre_result", "singer_sep")
                                                       .groupBy("singer_sep").agg(sum("pre_result") as "result") //sum("pre_result")*penality*5:to overcome the some singer too high according to hechang
                                                       .withColumnRenamed("singer_sep", "term")

    val df_singer_single = df_singer_single_pre.select("term", "result")
                                               .union(df_singer_multiple_sep.select("term", "result"))
                                               .groupBy("term").agg(sum("result") as "result")

    val df_singer = df_singer_single.select("term", "result")
                                    .union(df_singer_multiple)
                                    .withColumn("alias", myalias($"term"))
                                    .select("term", "result", "alias")

    val df_song = df_sn_sep.filter($"song".isNotNull && $"song" =!= "")
                           .groupBy("song").agg(sum("hot") as "result") //to burst the new song
                           .withColumnRenamed("song", "term")
                           .withColumn("alias", lit(""))
                           .select("term", "result", "alias")

    val df_songkind = df_sn_sep.filter($"kind".isNotNull && $"kind" =!= "")
                               .withColumn("kindfilter", lower($"kind"))
                               .filter($"kindfilter" =!= "live")
                               .withColumn("term",concat($"song", lit(" "), $"kind"))
                               .groupBy("term").agg(sum("hot")*penality as "result")
                               .withColumn("alias", lit(""))
                               .select("term", "result", "alias")

    val df_album = df_sn_sep.filter($"albumname".isNotNull && $"albumname" =!= "")
                            .filter(not($"albumname" rlike ".*原声[带]?$"))
                            .groupBy("albumname").agg(sum("hot")*penality*penality as "result")
                            .withColumnRenamed("albumname", "term")
                            .withColumn("alias", lit(""))
                            .select("term", "result", "alias")

    //generate index

    //6) union all related dataframe to df_final(created by song self)
    val df_term = df_singersong.union(df_songsinger)
                               .union(df_singer)
                               .union(df_song)
                               .union(df_songkind)
                               .union(df_album)
                               .groupBy("term", "alias").agg(max("result") as "result_temp")

    val df_final = df_term.select($"term", bround($"result_temp", 3) as "result", $"alias")

    df_final.persist()
    df_final.createOrReplaceTempView("term_savetable")
    val sql_term_save_create ="""
create table if not exists temp.jimmy_dt_hot_result_update
(
        term string,
        result DOUBLE,
        alias string
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_term_save_create)
    val sql_term_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_update PARTITION(cdt='$date_today', time='$update_time') select term, result, alias from term_savetable"
    spark.sql(sql_term_save)

    //7) compare with original df_triple
    //extract yesterday's data
    val sql_triple_read = s"select term, result, alias from temp.jimmy_dt_hot_result_mix where cdt='$date_end'"
    val df_triple_original = spark.sql(sql_triple_read)
    val df_triple_update = df_triple_original.select("term", "result", "alias")
                                             .union(df_final)
                                             .groupBy("term", "alias").agg(max("result") as "result")
                                             .select("term", "result", "alias")

    df_triple_update.createOrReplaceTempView("search_savetable")
    df_triple_update.persist()
    val sql_search_create ="""
create table if not exists temp.jimmy_dt_hot_result_mix_update
(
        term string,
        result DOUBLE,
        alias string
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_search_create)

    val sql_search = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_mix_update PARTITION(cdt='$date_today', time='$update_time') select term, result, alias from search_savetable"
    spark.sql(sql_search)

    //8)add num to df_search, then create df_search_num with term, result, resultnum, alias
    val sql_search_num = """
select
term,
result,
resultnum,
alias
from search_savetable a
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
)b on a.term = b.keyword
"""
    val df_search_num = spark.sql(sql_search_num)
    df_search_num.createOrReplaceTempView("num_search_savetable")
    val sql_result_num_create ="""
create table if not exists temp.jimmy_dt_hot_result_num_mix_update
(
        term string,
        result DOUBLE,
        resultnum INT,
        alias string
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_result_num_create)
    val sql_result_num = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_num_mix_update PARTITION(cdt='$date_today', time='$update_time') select * from num_search_savetable"
    spark.sql(sql_result_num)

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
