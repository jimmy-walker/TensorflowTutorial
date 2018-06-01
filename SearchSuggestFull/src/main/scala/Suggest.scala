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
object Suggest extends Serializable{

  def main(args: Array[String]):Unit = {
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Generate search suggest for yesterday")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._
    spark.conf.set("spark.sql.shuffle.partitions", 2001)

    //2)initial variable
    val date_end = args(0) //first argument is yesterday
    val hot_threshold = 30
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

    //4)read date
    //read remark data
    val sql_remark = s"select scid_albumid, remark, hot, type, rel_album_audio_id from temp.search_remark where cdt='$date_end'"
    val df_remark = spark.sql(sql_remark)
    df_remark.persist()
    //read seperated song data
    val sql_sn_sep_read = s"select mixsongid, albumname, song, kind, singer, hot from temp.jimmy_dt_sn_sep where cdt='$date_end'"
    val df_sn_sep = spark.sql(sql_sn_sep_read)

    //5)create related dataframe
    //generate remark related dataframe
    //===================================================
    //remark field without "", but we do it for safe purpose
    val df_remark_music = df_remark.filter($"type" === "2")
                                   .filter($"remark" =!= "")
                                   .select("scid_albumid", "remark", "hot")
    val df_remark_language = df_remark.filter($"type" === "8")
                                      .filter($"remark" =!= "")
                                      .select("scid_albumid", "remark", "hot")

    val df_remark_ref = df_remark.filter($"type" === "4" || $"type" === "5")
                                 .filter($"remark" =!= "")
                                 .filter($"rel_album_audio_id" =!= "0")
                                 .select("scid_albumid", "remark", "hot", "rel_album_audio_id")
                                 .withColumnRenamed("hot", "cover_hot")

    val df_remark_translate = df_remark.filter($"type" === "7")
                                       .filter($"remark".isNotNull && $"remark" =!= "")
                                       .select("scid_albumid", "remark", "hot")

    val df_remark_notion = df_remark_music.filter(not($"remark".contains("@BI_ROW_SPLIT@")))
                                          .withColumn("notion", regexp_extract($"remark", "《(.*)》", 1))
                                          .withColumn("category", myfunc(regexp_replace($"remark", "《.*》", "")))

    val df_remark_multiple = df_remark_music.filter($"remark".contains("@BI_ROW_SPLIT@"))
                                            .withColumn("remark_new", explode(split($"remark", "@BI_ROW_SPLIT@")))
                                            .select("scid_albumid", "hot", "remark_new")
                                            .withColumnRenamed("remark_new", "remark")
                                            .withColumn("notion", regexp_extract($"remark", "《(.*)》", 1))
                                            .withColumn("category", myfunc(regexp_replace($"remark", "《.*》", "")))

    val df_remak_final = df_remark_notion.select("scid_albumid", "hot", "remark", "notion", "category")
                                         .union(df_remark_multiple.select("scid_albumid", "hot", "remark", "notion", "category"))

    val df_remark_language_final = df_remark_language.filter(not($"remark".contains("@BI_ROW_SPLIT@")))
                                                     .withColumn("notion", regexp_extract($"remark", "《(.*)》", 1))
                                                     .withColumn("category", regexp_replace($"remark", "《.*》", ""))

    val df_remark_ref_cover = df_remark_ref.join(df_sn_sep, df_remark_ref("scid_albumid") === df_sn_sep("mixsongid"), "left")
                                           .select("scid_albumid", "cover_hot", "rel_album_audio_id", "song")
                                           .withColumnRenamed("song", "cover_song")
    val df_remark_ref_final = df_remark_ref_cover.join(df_sn_sep, df_remark_ref_cover("rel_album_audio_id") === df_sn_sep("mixsongid"), "left")
                                                 .filter($"hot".isNotNull)
                                                 .select("scid_albumid", "cover_hot", "rel_album_audio_id", "cover_song", "song", "hot")
    //===================================================

    //generate seperated song related dataframe
    //===================================================
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
                           .groupBy("song").agg(sum("hot") as "result")
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

    val df_notion = df_remak_final.filter($"notion".isNotNull && $"notion" =!= "")
                                  .groupBy("notion").agg(sum("hot") as "result")
                                  .withColumnRenamed("notion", "term")
                                  .withColumn("alias", lit(""))
                                  .select("term", "result", "alias")

    val df_notioncategory = df_remak_final.filter($"notion".isNotNull && $"notion" =!= "" && $"category".isNotNull && $"category" =!= "")
                                          .filter($"category" =!= "原声带" && $"category" =!= "原声")
                                          .withColumn("term",concat($"notion", lit(" "), $"category"))
                                          .groupBy("term").agg(sum("hot")*penality as "result")
                                          .withColumn("alias", lit(""))
                                          .select("term", "result", "alias")

    val df_language = df_remark_language_final.filter($"notion".isNotNull && $"notion" =!= "" && $"category".isNotNull && $"category" =!= "")
                                              .withColumn("term",concat($"notion", lit(" "), $"category"))
                                              .groupBy("term").agg(sum("hot")*penality as "result")
                                              .withColumn("alias", lit(""))
                                              .select("term", "result", "alias")

    val df_translate = df_remark_translate.groupBy("remark").agg(sum("hot") as "result")
                                          .withColumnRenamed("remark", "term")
                                          .withColumn("alias", lit(""))
                                          .select("term", "result", "alias")

    val window_cover_song = Window.partitionBy("cover_song")
    val window_song = Window.partitionBy("song")

    val df_ref_diff = df_remark_ref_final.filter($"song" =!= $"cover_song")
                                         .withColumn("cover_value", sum($"cover_hot").over(window_cover_song))
                                         .withColumn("value", max($"hot").over(window_song)) //change avg to max, cause it will exists many-many relation between song and hot, some hots are so small, it will lower the sum value, eg:凉凉
                                         .withColumn("term",concat($"cover_song", lit(" 原唱")))
                                         .withColumn("penality", $"value"*penality)
                                         .groupBy("term").agg(max("penality") as "result")
                                         .select("term", "result")

    val df_ref_eql = df_remark_ref_final.filter($"song" === $"cover_song")
                                        .withColumn("cover_value", sum($"cover_hot").over(window_cover_song))
                                        .withColumn("value", max($"hot").over(window_song)) //change avg to max, cause it will exists many-many relation between song and hot, some hots are so small, it will lower the sum value, eg:凉凉
                                        .withColumn("term",concat($"cover_song", lit(" 原唱")))
                                        .withColumn("penality", when($"cover_value" > $"value", $"value"*penality as "result").otherwise($"value"*penality*penality*penality as "result"))
                                        .groupBy("term").agg(max("penality") as "result")
                                        .select("term", "result")

    val df_ref = df_ref_eql.union(df_ref_diff)
                           .groupBy("term").agg(max("result") as "result")
                           .withColumn("alias", lit(""))
                           .select("term", "result","alias")//change the order otherwise it will change the column result from double to string
    //===================================================

    //6) union all related dataframe to df_final(created by song self)
    val df_term = df_singersong.union(df_songsinger)
                               .union(df_singer)
                               .union(df_song)
                               .union(df_songkind)
                               .union(df_album)
                               .union(df_notion)
                               .union(df_notioncategory)
                               .union(df_language)
                               .union(df_translate)
                               .union(df_ref)
                               .groupBy("term", "alias").agg(max("result") as "result")

    df_term.persist()

    //7)deal with manual input
    //read saved value
    //read kw and hot within three days
    val sql_kw_read= s"select dt, kw, scid_albumid, hot from temp.search_three_day_full where cdt = '$date_end'"
    val df_kw_dt = spark.sql(sql_kw_read)
    val df_kw_all = df_kw_dt.groupBy("kw").agg(sum("hot") as "hot")
                            .filter($"hot" > hot_threshold) //add the minimum threshold

    //standard the data, (data - mean)/std
    // use save value from yesterday
    val sql_read_value = s"select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from temp.jimmy_dt_save_value where cdt='$date_end'"
    val df_save_value = spark.sql(sql_read_value)
    val (hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min) = (df_save_value.first.getDouble(0), df_save_value.first.getDouble(1), df_save_value.first.getDouble(2), df_save_value.first.getDouble(3), df_save_value.first.getDouble(4), df_save_value.first.getDouble(5))

    //use saved values:hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min
    val df_kw_cal = df_kw_all.withColumn("hot_standard", ($"hot"-hot_mean)/hot_std)

    //we only use the positive part of hot_standard
    val df_kw_manual = df_kw_cal.withColumn("positive2", $"hot_standard" + lit(hot_min))
                                .filter($"positive2" > 0)
                                .select($"kw", bround($"positive2", 3) as "hot")

    //8) merge kw and df_final to df_mix with term, result, alias
    val df_kw_manual_strip = df_kw_manual.withColumn("origin2", mystrip($"kw"))
                                         .groupBy("origin2").agg(max("hot") as "hot")
    val df_final_strip = df_term.withColumn("origin", mystrip($"term"))

    val df_kw_manual_final = df_kw_manual_strip.join(df_final_strip, df_kw_manual_strip("origin2") === df_final_strip("origin"), "left")
    //list only manual exist term
    val df_kw_manual_unique = df_kw_manual_final.filter($"origin".isNull)
                                                .groupBy("origin2").agg(sum("hot") as "result") //to sum all the input by person
                                                .withColumnRenamed("origin2", "term")
                                                .withColumn("alias", myalias($"term"))
                                                .select("term", "result", "alias")
    //list both exist but manual is bigger than df_final
    val df_anti = df_kw_manual_final.filter($"hot" > $"result") //to overcome some small thing override
                                    .select("origin2", "hot", "term")
                                    .withColumnRenamed("term", "term_out")

    val df_final_exist = df_term.join(df_anti, df_term("term") === df_anti("term_out"), "leftanti")
                                 .select("term", "result", "alias")

    val df_kw_manual_exist = df_anti.groupBy("origin2").agg(max("hot") as "result")
                                    .withColumnRenamed("origin2", "term")
                                    .withColumn("alias", myalias($"term"))
                                    .select("term", "result", "alias")

    val df_mix = df_final_exist.union(df_kw_manual_exist)
                               .union(df_kw_manual_unique)
                               .groupBy("term", "alias").agg(max("result") as "result")
    df_mix.persist()


    //9)add the tag of variety to df_search with term, result, alias
    // for example chuangzao101
    val sql_kw_variety_album = s"""
select a.singername, a.albumid
from common.k_singer_album_part a
LEFT SEMI JOIN
(
select singerid
from common.k_singer_part
where dt='$date_end' and sextype = '12'
group by singerid
)b
on
(a.singerid = b.singerid
AND a.dt='$date_end')
"""
    val df_variety_album = spark.sql(sql_kw_variety_album)
    df_variety_album.createOrReplaceTempView("varitey_albumtable")


    val sql_kw_variety_song = s"""
select a.mixsongid, a.albumid
from common.st_k_mixsong_part a
LEFT SEMI JOIN
varitey_albumtable b
on
(a.albumid = b.albumid
AND a.dt='$date_end')
"""
    val df_variety_song = spark.sql(sql_kw_variety_song)
    //read
    val sql_scid_read=s"select scid_albumid, hot from temp.jimmy_dt_hot_score where cdt = '$date_end'"
    val df_scid = spark.sql(sql_scid_read)
    df_scid.createOrReplaceTempView("table_scid")

    val df_variety_temp = df_variety_album.join(df_variety_song, df_variety_album("albumid") === df_variety_song("albumid"))
                                          .drop("albumid") //it will remove two columns
    val df_variety = df_variety_temp.join(df_scid, df_variety_temp("mixsongid") === df_scid("scid_albumid"))
                                    .drop("mixsongid","scid_albumid","cdt")
                                    .groupBy("singername").agg(sum("hot")*penality as "result")
                                    .withColumnRenamed("singername", "term")
                                    .withColumn("alias", myalias($"term"))
                                    .select("term", "result", "alias")

    val df_search = df_mix.select("term", "result", "alias")
                          .union(df_variety)
                          .groupBy("term", "alias").agg(max("result") as "result")

    df_search.persist()

    //10) read from specified tag
    val sql_scid_hot = s"""
select a.scid_albumid, b.scid, a.hot
from
table_scid a
left join
common.st_k_mixsong_part b
on
(a.scid_albumid = b.mixsongid
AND b.dt='$date_end')
"""
    val df_scid_album_hot = spark.sql(sql_scid_hot)
    // there are some missing scid, it doesn't matter

    val df_scid_hot = df_scid_album_hot.filter($"scid".isNotNull)
                                       .groupBy("scid").agg(sum("hot") as "hot")

    df_scid_hot.createOrReplaceTempView("table_scid_hot")
    //there exist some missing for scid with tagid, remove the tagid is Null
    val sql_scid_tag_hot = s"""
select
        c.scid,
        c.tagid,
        d.tagname,
        c.hot
from (
        select
                a.scid,
                a.hot,
                b.tagid
        from table_scid_hot a
        left join common.k_sc_tag_part b
        on (
                a.scid = b.scid
                AND b.dt='$date_end'
        )
        where b.tagid IS NOT NULL
) c
left join common.k_tag_sc_part d
on (
        c.tagid = d.tagid
        AND d.dt='$date_end'
)
where d.tagname IS NOT NULL
"""
    val df_scid_tag_hot = spark.sql(sql_scid_tag_hot)
                               .groupBy("scid","hot").agg(concat_ws("-", collect_set("tagname")) as "tag")
    df_scid_tag_hot.createOrReplaceTempView("table_scid_tag_hot")
    df_scid_tag_hot.persist() //to save execute time by line 461

    //to avoid these table be removed by system
    val sql_tag_keyword_create = """
create table if not exists temp.jimmy_tag
(
tag string,
kw string
)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_tag_keyword_create)

    val sql_tag_keyword = """
select
        tag,
        kw
from temp.jimmy_tag
"""
    val df_tag_keyword = spark.sql(sql_tag_keyword)
    val tag_map = df_tag_keyword.select("tag","kw").as[(String, String)].collect.toMap

    var tag_hot_buf = new ListBuffer[(String, Double)]() //result is also double type
    for ((k,v) <- tag_map){ // k is the key, v is the value
      if(v != null & v != ""){ //to avoid null and empty string
        tag_hot_buf += ((k, df_scid_tag_hot.filter(v.split("-").toList.foldLeft(lit(true))((acc, x) => (acc && col("tag").contains(x)))).agg(sum("hot")).first.getAs[Double](0)))
      }
      else {
        tag_hot_buf += ((k, 0))
      }
    }
    val tag_hot = tag_hot_buf.toList
                             .toDF("term","result")
                             .withColumn("alias", lit(""))
                             .select("term", "result", "alias")

    val df_triple = df_search.select("term", "result", "alias")
                             .union(tag_hot)
                             .groupBy("term", "alias").agg(max("result") as "result")

    df_triple.createOrReplaceTempView("search_savetable")
    df_triple.persist()
    val sql_search_create= s"""
create table if not exists temp.jimmy_dt_hot_result_mix
(
term string,
result DOUBLE,
alias string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_search_create)

    val sql_search= s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_mix PARTITION(cdt='$date_end') select term, result, alias from search_savetable
"""
    spark.sql(sql_search)

    //11) add num to df_search, then create df_search_num with term, result, resultnum, alias
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
    val sql_result_num_create= s"""
create table if not exists temp.jimmy_dt_hot_result_num_mix
(
term string,
result DOUBLE,
resultnum INT,
alias string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    spark.sql(sql_result_num_create)

    val sql_result_num= s"""
INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_num_mix PARTITION(cdt='$date_end') select term, result, resultnum, alias from num_search_savetable
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
