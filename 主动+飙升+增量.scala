//1)import and setting
import scala.math.pow
import scala.math.abs
import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType //to solve the problem：Value toDF is not a member
import java.util.regex.Matcher
import java.util.regex.Pattern
spark.conf.set("spark.sql.shuffle.partitions", 2001)

//2)variable
//list variable
val date_end = "2018-05-18"
val update_time = "19"
val penality = 0.1
// val period = 3 // for only hot
val period = 14 // for burst and hot
val moving_average_length = 7
val hot_length = period - 1 //only calculate one day!!!!!
//filter_tag starts from 1, so hot_length will ensure the recent three days
val category = List("ost", "插曲", "主题曲", "原声带", "配乐", "片尾曲", 
                    "片头曲", "originalsoundtrack", "原声", "宣传曲", "op", 
                    "ed", "推广曲", "角色歌", "in", "背景音乐", "tm", "钢琴曲", 
                    "开场曲", "剧中曲", "bgm", "暖水曲", "主题歌")
val broadcasted = sc.broadcast(category)
val alias = Map("^G.E.M.邓紫棋$|^G.E.M.邓紫棋(?=、)|(?<=、)G.E.M.邓紫棋(?=、)|(?<=、)G.E.M.邓紫棋$" -> "邓紫棋")
val broadcasted_alias = sc.broadcast(alias)
//define date first
//list date function
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
val date_period = getDaysPeriod(date_end, period - 1)
val date_start = date_period.takeRight(1)(0)
val date_period_value = getDaysPeriod(date_end, 1) //take yesterday
val date_value = date_period_value.takeRight(1)(0)

case class TempRow(label: Int, dt: String)
var date_list_buffer = new ListBuffer[TempRow]()
for (dt <- date_period){
  date_list_buffer += TempRow(1, dt)
}
val date_list = date_list_buffer.toList

//2)-2 get df_date
//分段执行spark-shell
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
// //list date function
// def getDaysPeriod(dt: String, interval: Int): List[String] = {
//   var period = new ListBuffer[String]() //initialize the return List period
//   period += dt
//   val cal: Calendar = Calendar.getInstance() //reset the date in Calendar
//   cal.set(dt.split("-")(0).toInt, dt.split("-")(1).toInt - 1, dt.split("-")(2).toInt)
//   val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd") //format the output date
//   for (i <- 0 to interval - 1){
//     cal.add(Calendar.DATE, - 1)
//     period += dateFormat.format(cal.getTime())
//   }
//   period.toList
// }
// val date_period = getDaysPeriod(date_end, period - 1)
// val date_start = date_period.takeRight(1)(0)

//define window related function
def getWeight(length: Int): List[Double]= {
  var sum = 0.0
  for (i <- 0 to length-1 ){
    sum += math.pow(0.5, i)
  }
  val weights = for (i <- List.range(0, length) ) yield math.pow(0.5, i)/sum
  weights
}

def weighted_average(index: List[Int], weights: List[Double], w: WindowSpec, c: Column): Column= {
  val wma_list = for (i <- index) yield (lag(c, i).over(w))*weights(i) // list comprehension, map also can do some easy thing, return scala.collection.immutable.IndexedSeq
  wma_list.reduceLeft(_ + _)
}

//4)raw data sql and dataframe save
//calculate sql from three platform
//ard
//first calculate three days's data
//1.extract one day's dt,kw,scid_albumid,spt_cnt,status
val sql_ard_scid_one = """
select
        dt,
        scid_albumid,
        hot
from (
        select 
                dt,
                scid_albumid,
                sum(case when (status='完整播放' or spt_cnt>=30) then 1 else 0 end) as hot
        from (
                select 
                        dt,
                        COALESCE(ROUND((CASE WHEN (
                                                    CASE WHEN tv<7900 AND spt<0 THEN 0
                                                         WHEN tv<7900 AND spt>=0 THEN spt
                                                         WHEN tv>=7900 AND ivar2<0 THEN 0
                                                         WHEN tv>=7900 AND ivar2>=0 THEN ivar2
                                                    ELSE 0 END)>50*st THEN st
                                        ELSE (CASE WHEN tv<7900 AND spt<0 THEN 0
                                                   WHEN tv<7900 AND spt>=0 THEN spt
                                                   WHEN tv>=7900 AND ivar2<0 THEN 0
                                                   WHEN tv>=7900 AND ivar2>=0 THEN ivar2
                                                   ELSE 0 END
                                        ) END)/1000,6),
                        '0') as spt_cnt,
                        status,
                        scid_albumid  
                from (
                        select 
                                dt,
                                CASE WHEN CAST(tv1 AS INT) IS NOT NULL THEN tv1 ELSE COALESCE(CAST(tv AS INT),'unknown') END AS tv,
                                CASE WHEN TRIM(sty)='音频' THEN '音频'
                                WHEN TRIM(sty)='视频' THEN '视频'
                                ELSE 'unknown' END sty,
                                CASE WHEN TRIM(fs)='完整播放' THEN '完整播放'
                                WHEN TRIM(fs)='播放错误' THEN '播放错误'
                                WHEN TRIM(fs) IN ('被终止','播放中退出','暂停时退出','被終止') THEN '未完整播放'
                                ELSE '未知播放状态' END status,
                                case when abs(coalesce(cast(st as decimal(20,0)),0))>=10000000 then abs(coalesce(cast(st as decimal(20,0)),0))/1000 else abs(coalesce(cast(st as decimal(20,0)),0)) end  st,
                                case when coalesce(cast(spt as decimal(20,0)),0)>=10000000 then coalesce(cast(spt as decimal(20,0)),0)/1000 else coalesce(cast(spt as decimal(20,0)),0) end spt,
                                COALESCE(CAST(ivar2 AS decimal(20,0)),0) ivar2,
                                trim(scid_albumid) scid_albumid
                        from ddl.dt_list_ard_d
                        WHERE dt = """+s"""'$date_end'"""+"""
                                and action='play'
                                and TRIM(fs)<>'播放错误'
                                and sh <>'00000000000000000000000000000000'
                                and st is not null
                                and st<>'null'
                                and ivar10 = '主动播放'
                                and TRIM(scid_albumid)<>'0'
                                and (fo like '%搜索/%' or fo='搜索') and (fo not like '%本地音乐/%')
                )t
                where sty='音频'
        )a
        group by dt, scid_albumid
)b where hot>0
"""

val df_ard_day_one = spark.sql(sql_ard_scid_one)
df_ard_day_one.persist()
//2.get the distinct scid within three days to be reached in 14 days after
//select kw and scid_albumid seperatively
//one is df_ard_scid for 14 days
//other is df_ard_day_three for 3 days

val df_ard_day_scid_unique = df_ard_day_one.select("scid_albumid")
df_ard_day_scid_unique.persist()
df_ard_day_scid_unique.createOrReplaceTempView("table_ard_day_scid")

//3.extract period day's dt,scid_albumid,spt_cnt,status 
val sql_ard_scid_period = """
select 
        dt,
        scid_albumid,
        sum(case when (status='完整播放' or spt_cnt>=30) then 1 else 0 end) as hot
from (
        select 
                dt,
                COALESCE(ROUND((CASE WHEN (
                                            CASE WHEN tv<7900 AND spt<0 THEN 0
                                                 WHEN tv<7900 AND spt>=0 THEN spt
                                                 WHEN tv>=7900 AND ivar2<0 THEN 0
                                                 WHEN tv>=7900 AND ivar2>=0 THEN ivar2
                                            ELSE 0 END)>50*st THEN st
                                ELSE (CASE WHEN tv<7900 AND spt<0 THEN 0
                                           WHEN tv<7900 AND spt>=0 THEN spt
                                           WHEN tv>=7900 AND ivar2<0 THEN 0
                                           WHEN tv>=7900 AND ivar2>=0 THEN ivar2
                                           ELSE 0 END
                                ) END)/1000,6),
                '0') as spt_cnt,
                status,
                scid_albumid  
        from (
                select 
                        dt,
                        CASE WHEN CAST(tv1 AS INT) IS NOT NULL THEN tv1 ELSE COALESCE(CAST(tv AS INT),'unknown') END AS tv,
                        CASE WHEN TRIM(sty)='音频' THEN '音频'
                        WHEN TRIM(sty)='视频' THEN '视频'
                        ELSE 'unknown' END sty,
                        CASE WHEN TRIM(fs)='完整播放' THEN '完整播放'
                        WHEN TRIM(fs)='播放错误' THEN '播放错误'
                        WHEN TRIM(fs) IN ('被终止','播放中退出','暂停时退出','被終止') THEN '未完整播放'
                        ELSE '未知播放状态' END status,
                        case when abs(coalesce(cast(st as decimal(20,0)),0))>=10000000 then abs(coalesce(cast(st as decimal(20,0)),0))/1000 else abs(coalesce(cast(st as decimal(20,0)),0)) end  st,
                        case when coalesce(cast(spt as decimal(20,0)),0)>=10000000 then coalesce(cast(spt as decimal(20,0)),0)/1000 else coalesce(cast(spt as decimal(20,0)),0) end spt,
                        COALESCE(CAST(ivar2 AS decimal(20,0)),0) ivar2,
                        trim(scid_albumid) scid_albumid
                from ddl.dt_list_ard_d e LEFT SEMI JOIN table_ard_day_scid f on (e.scid_albumid = f.scid_albumid)
                WHERE dt BETWEEN """+s"""'$date_start'"""+""" AND """+s"""'$date_value'"""+"""
                        and action='play'
                        and TRIM(fs)<>'播放错误'
                        and sh <>'00000000000000000000000000000000'
                        and st is not null
                        and st<>'null'
                        and ivar10 = '主动播放'
                        and TRIM(scid_albumid)<>'0'
                        and (fo like '%搜索/%' or fo='搜索') and (fo not like '%本地音乐/%')
        )t
        where sty='音频'
)a
group by dt, scid_albumid
"""

val df_ard_scid_period = spark.sql(sql_ard_scid_period)

val df_ard_scid = df_ard_scid_period.select("dt", "scid_albumid", "hot")
                                    .union(df_ard_day_one.select("dt", "scid_albumid", "hot"))
df_ard_scid.persist()

//save
df_ard_scid.createOrReplaceTempView("table_ard_scid")
val sql_ard_filter_scid = s"INSERT OVERWRITE TABLE temp.jimmy_scid_dt_ard_update_zhudong PARTITION(cdt='$date_end', time='$update_time') select dt, scid_albumid, hot from table_ard_scid"
spark.sql(sql_ard_filter_scid)
// //read
// val sql_ard_filter_scid_read = s"select dt, scid_albumid, hot from temp.jimmy_scid_dt_ard_update_zhudong where cdt='$date_end'"
// val df_ard_scid = spark.sql(sql_ard_filter_scid_read)

//above sql is useful
// scala> df_ard_scid.select("scid_albumid").distinct.count()
// res10: Long = 1002524                                                           
// scala> df_ard_day_scid_unique.count()
// res11: Long = 1002524  

//5)merge all and groupby to df_all with scid_albumid,hot,dt,label
//then fulfill the missing dt
//from df_all to generate df_period with complete dt
//spc_cnt is string, it will cast into 
val df_all = df_ard_scid.withColumn("label", lit(1))

df_all.createOrReplaceTempView("table_all")

val sql_period  = "select scid_albumid, a.dt, (case when a.dt=b.dt then hot else 0 end) as hot from date_table a RIGHT OUTER JOIN table_all b on a.label=b.label"
val df_period_temp = spark.sql(sql_period)
// to eliminate the extra data because of RIGHT OUTER JOIN
val df_period = df_period_temp.groupBy("dt","scid_albumid").agg(max("hot").as("hot"))
//df_period.persist()
// df_period.count()
// res5: Long = 42681562

//5)-2calculate the burst score
//generate df_cal_save with scid_albumid, burst, hot
//generate stanard output df_scid with scid_albumid, hot
//save values:hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min
//use window function
val weights = getWeight(moving_average_length)
val index = List.range(0,moving_average_length)
//calculate the weighted moving average
@transient val window_ma = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
@transient val window_ma_column = weighted_average(index, weights, window_ma, df_period("hot"))
val period_ma = df_period.withColumn("weightedmovingAvg", window_ma_column)
//filter the last n rows, cause their weighted moving average we didn't calculate, cause data is missing
@transient val window_filter = Window.partitionBy("scid_albumid").orderBy(asc("dt"))
@transient val window_filter_column = row_number.over(window_filter)
val period_filter_temp = period_ma.withColumn("filter_tag", window_filter_column)
val period_filter = period_filter_temp.filter($"filter_tag" >= moving_average_length)
//calculate the amp score
val period_burst = period_filter.groupBy("scid_albumid").agg((last("weightedmovingAvg")-(mean("weightedmovingAvg") + lit(1.5)*stddev_samp("weightedmovingAvg"))).as("weightedamp"))

//calculate the recent three days total hot
//filter_tag starts from 1, so hot_length will ensure the recent one days
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
df_cal_save.createOrReplaceTempView("cal_savetable")
val sql_burst_hot_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_burst_hot_score_update PARTITION(cdt='$date_end', time='$update_time') select scid_albumid, burst, hot from cal_savetable"
spark.sql(sql_burst_hot_save)

//use save value from yesterday
val sql_read_value = s"select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from temp.jimmy_dt_save_value where cdt='$date_value'"
val df_save_value = spark.sql(sql_read_value)
val (hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min) = (df_save_value.first.getDouble(0), df_save_value.first.getDouble(1), df_save_value.first.getDouble(2), df_save_value.first.getDouble(3), df_save_value.first.getDouble(4), df_save_value.first.getDouble(5))

//standard the data, (data - mean)/std
//we only use the burst score and one day hot score, cause we ignore the hot score it's meaningless to calculate the recent three day in update operation
val df_cal = df_cal_save.filter($"burst" > 0) //only extract the positive one
                        .withColumn("burst_standard", ($"burst"-burst_mean)/burst_std)
                        .withColumn("hot_standard", ($"hot"-hot_mean)/hot_std)

// scala> df_cal.count()
// res31: Long = 47844

//we only use the positive part of burst_standard
//we filter it above zero
//save both burst and hot
val df_temp = df_cal.withColumn("positive", when($"burst_standard" > lit(0), $"burst_standard").otherwise(lit(0)))
                    .withColumn("positive2", $"hot_standard" + lit(hot_min))
                    .filter($"positive" > 0) //we only calculate the bursting song
// val df_scid = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive" + $"positive2" + lit(1)).otherwise($"positive" + $"positive2" + lit(1)))
// we consier not to add lit(1) as bias
val df_scid_update = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive"  + $"positive2").otherwise($"positive" + $"positive2"))
                            .select($"scid_albumid", bround($"result_temp", 3) as "hot")  


//only save burst
// val df_temp = df_cal.withColumn("positive", when($"burst_standard" > lit(0), $"burst_standard").otherwise(lit(0)))
//                     .filter($"positive" > 0) //we only calculate the bursting song

// val df_scid_update = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive"  + $"positive2").otherwise($"positive" + $"positive2"))
//                             .select($"scid_albumid", bround($"result_temp", 3) as "hot")  


df_scid_update.persist()
df_scid_update.createOrReplaceTempView("table_scid_update")
val sql_scid_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_score_update PARTITION(cdt='$date_end', time='$update_time') select scid_albumid, hot from table_scid_update"
spark.sql(sql_scid_save)
// scala> df_scid_update.count()
// res2: Long = 47844 

//5)-3 filter the overriding scid_albumid, i.e. hot higher than yesterday
//read from yesterday's scid_albumid result
val sql_scid_original = s"select scid_albumid, hot from temp.jimmy_dt_hot_score where cdt='$date_value'"
val df_scid_original = spark.sql(sql_scid_original)

val df_scid_update_renamed = df_scid_update.withColumnRenamed("scid_albumid", "scid_albumid_update")
                                           .withColumnRenamed("hot", "hot_update")

val df_scid_update_final = df_scid_update_renamed.join(df_scid_original, df_scid_update_renamed("scid_albumid_update") === df_scid_original("scid_albumid"), "left")
//list only update exist term
val df_scid_update_unique = df_scid_update_final.filter($"hot".isNull)
                                                .select("scid_albumid_update", "hot_update")

//list both exist but update is bigger than original
val df_anti = df_scid_update_final.filter($"hot_update" > $"hot") //to overcome some small thing override
                                  .select("scid_albumid_update", "hot_update")

val df_scid = df_scid_update_unique.union(df_anti)
                                   .withColumnRenamed("scid_albumid_update", "scid_albumid")
                                   .withColumn("hot", $"hot_update"/penality)
                                   .select("scid_albumid", "hot")

df_scid.persist()
df_scid.createOrReplaceTempView("table_scid")
val sql_scid_new_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_score_new_update PARTITION(cdt='$date_end', time='$update_time') select scid_albumid, hot from table_scid"
spark.sql(sql_scid_new_save)

// save burst and hot
// scala> df_scid.count()
// res9: Long = 37224  

//==========================================================================


//6)using df_scid to extract name from common.st_k_mixsong not common.st_k_mixsong_part!!!!!
//select singername, songname, albumname

// //read
// val sql_scid_new_read = s"select scid_albumid, hot from temp.jimmy_dt_hot_score_new_update where cdt='$date_end'"
// val df_scid = spark.sql(sql_scid_new_read)


//first run the hive command!!!!!!!!!!!

val sql_sn = s"select mixsongid, choric_singer, songname, albumname, hot from temp.jimmy_dt_sn_update where cdt='$date_end'"
val df_sn = spark.sql(sql_sn)
//filter and format name
val df_sn_sep = df_sn.withColumn("song", regexp_replace($"songname", "[ ]*\\([^\\(\\)]*\\)$", ""))
                     .withColumn("kind", regexp_extract($"songname", "\\(([^\\(\\)]*)\\)$", 1))
                     .withColumnRenamed("choric_singer", "singer") //to eliminate the space effect

df_sn_sep.persist()
//there are exist some missing
// df_sn_sep.count()
// res1: Long = 37144
df_sn_sep.createOrReplaceTempView("table_sep")
val sql_sn_sep_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_sn_sep_update PARTITION(cdt='$date_end', time='$update_time') select mixsongid, albumname, song, kind, singer, hot from table_sep"
//remember that we should select the item in the hive item order!!!!!!!!!!!!!!!
spark.sql(sql_sn_sep_save)

// //read
// val sql_sn= s"select mixsongid, albumname, song, kind, singer, hot from temp.jimmy_dt_sn_sep_update where cdt='$date_end'"
// val df_sn_sep = spark.sql(sql_sn)
// df_sn_sep.persist()

////7)read remark and generate remark related dataframe


//8) generate index
//====================================================
//in fact we find exists phenomenon, it starts with " " space due to wrong input.
//but we decide to omit it. cause it will remove from the research engine.
//we have checked the df_sn_sep's singer, song, kind, albumname. all four without null, with ""; 
// cause we create these three columns, so it won't be null
// but we should omit the "" empty string
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
                                                                               
//generate index

//9) union all related dataframe to df_final(created by song self)
val df_term = df_singersong.union(df_songsinger)
                           .union(df_singer)
                           .union(df_song)
                           .union(df_songkind)
                           .union(df_album)
                           .groupBy("term", "alias").agg(max("result") as "result_temp")


val df_final = df_term.select($"term", bround($"result_temp", 3) as "result", $"alias")

df_final.persist()

df_final.createOrReplaceTempView("term_savetable")
val sql_term_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_update PARTITION(cdt='$date_end', time='$update_time') select term, result, alias from term_savetable"
spark.sql(sql_term_save)

//9)-2 compare with original df_triple
//extract yesterday's data
val sql_triple_read = s"select term, result, alias from temp.jimmy_dt_hot_result_mix where cdt='$date_value'"
val df_triple_original = spark.sql(sql_triple_read)

val df_triple_update = df_triple_original.select("term", "result", "alias")
                                         .union(df_final)
                                         .groupBy("term", "alias").agg(max("result") as "result")
                                         .select("term", "result", "alias")

df_triple_update.createOrReplaceTempView("search_savetable")
df_triple_update.persist()
val sql_search = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_mix_update PARTITION(cdt='$date_end', time='$update_time') select term, result, alias from search_savetable"
spark.sql(sql_search)

//====================================================================
//check the result
// df_final.filter(col("term").startsWith("起风了")).sort($"result".desc).show()

//10)manual input


//11) merge kw and df_final to df_mix with term, result, alias


//12) add the tag of variety to df_search with term, result, alias


//13) read from specified tag


//14) add num to df_search, then create df_search_num with term, result, resultnum, alias


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
where dt="""+s"""'$date_value'"""+"""
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
val sql_result_num = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_num_mix_update PARTITION(cdt='$date_end', time='$update_time') select * from num_search_savetable"
spark.sql(sql_result_num)