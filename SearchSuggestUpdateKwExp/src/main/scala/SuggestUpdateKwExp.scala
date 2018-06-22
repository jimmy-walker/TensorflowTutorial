/**
  *
  * author: jomei
  * date: 2018/6/22 17:19
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
class SuggestUpdateKwExp extends Serializable{
  def main(args: Array[String]): Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Generate index for updated scid_albumid")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)variable
    val date_today = args(0)
    val date_end = args(1)
    val update_time = args(2)
    val penality = 0.1

    val date_period = getDaysPeriod(date_end, 2) //take three days record
    val date_start = date_period.takeRight(1)(0)
    var date_list_buffer = new ListBuffer[TempRow]()
    for (dt <- date_period){
      date_list_buffer += TempRow(1, dt)
    }
    val date_list = date_list_buffer.toList
    val df_date = date_list.toDF
    df_date.createOrReplaceTempView("date_table")

    //3)raw prior data from recent three days
    val sql_scid_prior = s"select dt, scid_albumid, hot from temp.search_three_day_bayes_scid where cdt='$date_end'"
    val df_scid_prior = spark.sql(sql_scid_prior)
    df_scid_prior.persist()

    //4)merge all and groupBy to df_all with dt, scid_albumid, hot, label
    //then fulfill the missing dt
    //from df_all to generate df_period with complete dt
    val df_all = df_scid_prior.withColumn("label", lit(1))

    df_all.createOrReplaceTempView("table_all")

    val sql_prior  = "select scid_albumid, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
    val df_prior_temp = spark.sql(sql_prior)
    // to eliminate the extra data because of RIGHT OUTER JOIN
    val df_prior = df_prior_temp.groupBy("dt","scid_albumid").agg(max("hot").as("hot"))
    df_prior.persist()
    // df_prior.filter($"scid_albumid" === "53187182").show()

    //5)calculate the prior score:hot_new for scid_albumid
    //use window function
//    val weights = List(1.0, 0.1, 0.01)
    val weights = List(args(3).toDouble, args(4).toDouble, args(5).toDouble)
    val index = List.range(0,3)
    val window_sum = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
    val prior_sum = df_prior.withColumn("prior", weighted_average(index, weights, window_sum, df_prior("hot")))
    //filter the last n rows, cause their weighted moving average we didn't calculate, cause data is missing
    val window_filter = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
    val prior_filter_temp =prior_sum.withColumn("filter_tag", row_number.over(window_filter))
    prior_filter_temp.persist()
    val prior_filter = prior_filter_temp.filter($"filter_tag" >= 3)
                                        .filter($"prior" > 0)
                                        .filter($"scid_albumid".isNotNull && $"scid_albumid" =!= "" && $"scid_albumid" =!= "0")
                                        .withColumn("hot_new", bround($"prior", 2))
    prior_filter.persist()
    prior_filter.createOrReplaceTempView("table_scid_prior")

    //6) combine singerid with prior_filter, generate dataframe with album_audio_id, hot_prior
    // first extract recent three day's all singerid and scid_albumid pair
    val sql_scid_singerid = s"select album_audio_id, singerid from temp.jimmy_scid_singerid where cdt='$date_end'"
    val df_scid_singerid = spark.sql(sql_scid_singerid)
    df_scid_singerid.createOrReplaceTempView("table_scid_singerid")
    // second extract today's update singerid and update album_audio_id
    val sql_scid_singerid_today = s"select album_audio_id, author_id from temp.jimmy_dt_three_singer_kw_update where cdt='$date_today' and time='$update_time'"
    val df_scid_singerid_today = spark.sql(sql_scid_singerid_today)
    df_scid_singerid_today.persist() //it will be used later
    df_scid_singerid_today.createOrReplaceTempView("table_scid_singerid_today")
    // third only extract today's update singerid's all existed album_audio_id
    val df_scid_singerid_new = df_scid_singerid_today.select("author_id") //to eliminate the same singerid to increase the number
                                                     .distinct()
                                                     .as("d1")
                                                     .join(df_scid_singerid.as("d2"), $"d1.author_id" === $"d2.singerid", "left")
                                                     .select($"d2.album_audio_id", $"d2.singerid")
    df_scid_singerid_new.createOrReplaceTempView("table_scid_singerid_new")
    // forth calculate today's update singerid's max prior hot_prior
    val df_prior_combine = df_scid_singerid_new.as("d1")
                                               .join(prior_filter.as("d2"), $"d1.album_audio_id" === $"d2.scid_albumid", "left")
                                               .select($"d1.*", $"d2.hot_new")
    // df_prior_combine.filter($"singerid" === "5833").sort($"hot_new".desc).show()
    // df_prior_combine.filter($"singerid" === "6004").sort($"hot_new".desc).show()

    val df_singer_prior = df_prior_combine.groupBy("singerid").agg(max("hot_new") as "hot_prior")

    // fifth calculate today's update album_audio_id's max prior hot_prior
    val df_scid_albumid_prior = df_scid_singerid_today.as("d1")
                                                      .join(df_singer_prior.as("d2"), $"d1.author_id" === $"d2.singerid", "left")
                                                      .select($"d1.*", $"d2.hot_prior")
                                                      .na.fill(0) //to avoid singer without prior
                                                      .groupBy("album_audio_id").agg(max("hot_prior") as "hot_prior")

    //7) combine today's update scid_albumid's update_time's hot and hot_prior
    val sql_scid_hot = s"select scid_albumid, hot from temp.jimmy_dt_three_hot_score_kw_update_scid where cdt='$date_today' and time='$update_time'"
    val df_scid_hot = spark.sql(sql_scid_hot)

    val df_combine = df_scid_hot.as("d1")
                                .join(df_scid_albumid_prior.as("d2"), $"d1.scid_albumid" === $"d2.album_audio_id", "left")
                                .select($"d1.scid_albumid", $"d1.hot", $"d2.hot_prior")
                                .na.fill(0) // to avoid some mistakes
    df_combine.persist()
    df_combine.createOrReplaceTempView("table_prior_hot")
    // save two score
    val sql_prior_hot_create = """
create table if not exists temp.jimmy_dt_three_prior_hot
(
    scid_albumid string,
    hot DOUBLE,
    hot_prior DOUBLE
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_prior_hot_create)

    val sql_prior_hot = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_three_prior_hot PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, hot, hot_prior from table_prior_hot
"""
    spark.sql(sql_prior_hot)

    // save the sum score
    val df_score = df_combine.withColumn("score", bround($"hot" + $"hot_prior", 2))
                             .select("scid_albumid", "score")
    df_score.persist()
    df_score.createOrReplaceTempView("table_score")

    val sql_score_create = """
create table if not exists temp.jimmy_dt_three_score_update
(
    scid_albumid string,
    score DOUBLE
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_score_create)

    val sql_score = s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_three_score_update PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, score from table_score
"""
    spark.sql(sql_score)

    //8) read new song info(df_sn_sep) and save score and songname
    val sql_sn = s"select mixsongid, choric_singer, songname, albumname, hot from temp.jimmy_dt_three_sn_kw_update_scid where cdt='$date_today'"
    val df_sn = spark.sql(sql_sn)
    //filter and format name
    val df_sn_sep = df_sn.withColumn("song", regexp_replace($"songname", "[ ]*\\([^\\(\\)]*\\)$", ""))
                         .withColumn("kind", regexp_extract($"songname", "\\(([^\\(\\)]*)\\)$", 1))
                         .withColumn("singer", regexp_replace($"choric_singer", "[ ]*\\([0-9]*\\)$", "")) //to eliminate the duplicate singer effect

    df_sn_sep.persist()

    //save score and its songname
    val df_score_name = df_score.as("d1")
                                .join(df_sn_sep.as("d2"), $"d1.scid_albumid" === $"d2.mixsongid", "left")
                                .select($"d1.*", $"d2.song", $"d2.singer", $"d2.kind", $"d2.albumname")
    df_score_name.persist()
    df_score_name.createOrReplaceTempView("table_score_name")
    val sql_score_name_save_create ="""
create table if not exists temp.jimmy_dt_score_name_kw_update_scid
(
scid_albumid string,
score DOUBLE,
song string,
singer string,
kind string,
albumname string
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_score_name_save_create)
    val sql_score_name_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_score_name_kw_update_scid PARTITION(cdt='$date_today', time='$update_time') select scid_albumid, score, song, singer, kind, albumname from table_score_name"
    spark.sql(sql_score_name_save)

    // 9)generate index
    val df_singersong = df_score_name.filter($"singer".isNotNull && $"singer" =!= "" && $"song".isNotNull && $"song" =!= "")
                                     .groupBy("singer", "song").agg(sum("score")*penality as "result")
                                     .withColumn("term",concat($"singer", lit(" "), $"song"))
                                     .select("term", "result")

    val df_songsinger = df_score_name.filter($"singer".isNotNull && $"singer" =!= "" && $"song".isNotNull && $"song" =!= "")
                                     .groupBy("singer", "song").agg(sum("score")*penality as "result")
                                     .withColumn("term",concat($"song", lit(" "), $"singer"))
                                     .select("term", "result")

    val df_singer_single_pre = df_score_name.filter($"singer".isNotNull && $"singer" =!= "")
                                            .filter(not($"singer".contains("、")))
                                            .groupBy("singer").agg(sum("score") as "result")
                                            .withColumnRenamed("singer", "term")

    val df_singer_multiple_pre = df_score_name.filter($"singer".isNotNull && $"singer" =!= "")
                                              .filter($"singer".contains("、"))
                                              .groupBy("singer").agg(sum("score") as "pre_result")
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
                                    .select("term", "result")

    val df_song = df_score_name.filter($"song".isNotNull && $"song" =!= "")
                               .groupBy("song").agg(sum("score") as "result") //to burst the new song
                               .withColumnRenamed("song", "term")
                               .select("term", "result")

    val df_songkind = df_score_name.filter($"kind".isNotNull && $"kind" =!= "")
                                   .withColumn("kindfilter", lower($"kind"))
                                   .filter($"kindfilter" =!= "live")
                                   .withColumn("term",concat($"song", lit(" "), $"kind"))
                                   .groupBy("term").agg(sum("score")*penality as "result")
                                   .select("term", "result")

    val df_album = df_score_name.filter($"albumname".isNotNull && $"albumname" =!= "")
                                .filter(not($"albumname" rlike ".*原声[带]?$"))
                                .groupBy("albumname").agg(sum("score")*penality*penality as "result")
                                .withColumnRenamed("albumname", "term")
                                .select("term", "result")
    //generate index

    //10) union all related dataframe to df_final(created by song self)
    val df_term = df_singersong.union(df_songsinger)
                               .union(df_singer)
                               .union(df_song)
                               .union(df_songkind)
                               .union(df_album)
                               .groupBy("term").agg(max("result") as "result_temp")

    val df_final = df_term.select($"term", bround($"result_temp", 3) as "hot")
                          .withColumnRenamed("term", "kw")

    df_final.persist()
    df_final.createOrReplaceTempView("term_savetable")
    val sql_term_save_create ="""
create table if not exists temp.jimmy_dt_three_hot_result_kw_update_scid
(
kw string,
hot DOUBLE
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_term_save_create)
    val sql_term_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_three_hot_result_kw_update_scid PARTITION(cdt='$date_today', time='$update_time') select kw, hot from term_savetable"
    spark.sql(sql_term_save)

    //11) compare with original df_triple
    //extract yesterday's data
    val sql_triple_read = s"select kw, hot from temp.jimmy_dt_three_hot_score_kw where cdt='$date_end'"
    val df_triple_original = spark.sql(sql_triple_read)
    val df_triple_update = df_triple_original.select("kw", "hot")
                                             .union(df_final.select("kw", "hot"))
                                             .groupBy("kw").agg(max("hot") as "hot")
                                             .select("kw", "hot")

    df_triple_update.createOrReplaceTempView("search_savetable")
    df_triple_update.persist()
    val sql_search_create ="""
create table if not exists temp.jimmy_dt_three_hot_result_mix_kw_update_scid
(
kw string,
hot DOUBLE
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_search_create)

    val sql_search = s"INSERT OVERWRITE TABLE temp.jimmy_dt_three_hot_result_mix_kw_update_scid PARTITION(cdt='$date_today', time='$update_time') select kw, hot from search_savetable"
    spark.sql(sql_search)

    //12)add num to df_search, then create df_search_num with term, result, resultnum, alias
    val sql_search_num = """
select
kw,
hot,
resultnum
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
)b on a.kw = b.keyword
"""
    val df_search_num = spark.sql(sql_search_num)
    df_search_num.createOrReplaceTempView("num_search_savetable")
    val sql_result_num_create ="""
create table if not exists temp.jimmy_dt_three_hot_result_num_mix_kw_update_scid
(
kw string,
hot DOUBLE,
resultnum INT
)
partitioned by (cdt string, time string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_result_num_create)
    val sql_result_num = s"INSERT OVERWRITE TABLE temp.jimmy_dt_three_hot_result_num_mix_kw_update_scid PARTITION(cdt='$date_today', time='$update_time') select * from num_search_savetable"
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
