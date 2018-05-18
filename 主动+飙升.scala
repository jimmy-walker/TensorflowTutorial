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
val date_end = "2018-05-17"
val penality = 0.1
// val period = 3 // for only hot
val period = 14 // for burst and hot
val moving_average_length = 7
val hot_length = period - 2
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
val date_period_filter = getDaysPeriod(date_end, 2) //take three days record
val date_filter = date_period_filter.takeRight(1)(0)
val date_period_filter_before = getDaysPeriod(date_end, 3) //take forth days record
val date_filter_before = date_period_filter_before.takeRight(1)(0)


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
//1.extract three day's dt,kw,scid_albumid,spt_cnt,status
val sql_ard_day_three = """
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
        split(fo,'/')[2] as kw,
        status,
        scid_albumid  
from (
        select 
                dt,
                CASE WHEN CAST(tv1 AS INT) IS NOT NULL THEN tv1 ELSE COALESCE(CAST(tv AS INT),'unknown') END AS tv,
                COALESCE(TRIM(fo),'unknown') fo,
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
                scid_albumid
        from ddl.dt_list_ard_d
        WHERE dt BETWEEN """+s"""'$date_filter'"""+""" AND """+s"""'$date_end'"""+"""
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
"""
val df_ard_day_three = spark.sql(sql_ard_day_three)
// df_ard_day_three.persist()
// df_ard_day_three.createOrReplaceTempView("table_ard_day_three")

//2.get the distinct scid within three days to be reached in 14 days after
//select kw and scid_albumid seperatively
//one is df_ard_scid for 14 days
//other is df_ard_day_three for 3 days

val df_ard_day_scid_temp = df_ard_day_three.withColumn("hot", when(($"status" === "完整播放") || ($"spt_cnt" >= 30), 1).otherwise(0))
                                           .select("dt", "kw", "scid_albumid", "hot")
df_ard_day_scid_temp.persist() //cause we will generate the scid_kw seperatively, then unpersist it

val df_ard_day_scid_unique = df_ard_day_scid_temp.select("scid_albumid", "hot")
                                                 .groupBy("scid_albumid").agg(sum("hot") as "hot")
                                                 .filter($"hot" > 0)
                                                 .select("scid_albumid")


df_ard_day_scid_unique.persist() //after all sql operation, later we unpersist it
df_ard_day_scid_unique.createOrReplaceTempView("table_ard_day_scid")


//3.first extract the kw only without burst
val df_ard_day_all = df_ard_day_scid_temp.as("d1")
                                         .join(df_ard_day_scid_unique.as("d2"), $"d1.scid_albumid" === $"d2.scid_albumid")
                                         .select($"d1.*")

df_ard_day_all.persist() //after all sql operation, later we unpersist it

val df_ard_day_all_kw = df_ard_day_all.groupBy("kw").agg(sum("hot") as "hot")
df_ard_day_all_kw.createOrReplaceTempView("table_ard_day_all_kw")
//save
val sql_ard_filter_kw = s"INSERT OVERWRITE TABLE temp.jimmy_kw_hot_ard_zhudong PARTITION(cdt='$date_end') select kw, hot from table_ard_day_all_kw"
spark.sql(sql_ard_filter_kw)
//consier delete save=====================================================

//read
// val sql_ard_filter_kw_read = s"select dt, kw, scid_albumid, spt_cnt, status from temp.jimmy_kw_dt_ard where cdt='$date_end'"
// val df_ard_day_three = spark.sql(sql_ard_filter_kw_read)
// df_ard_day_three.persist()
// df_ard_day_three.createOrReplaceTempView("table_ard_day_three")

//4.extract period day's dt,kw,scid_albumid,spt_cnt,status 
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
                WHERE dt BETWEEN """+s"""'$date_start'"""+""" AND """+s"""'$date_filter_before'"""+"""
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

val df_ard_day_all_scid = df_ard_day_all.groupBy("dt", "scid_albumid").agg(sum("hot") as "hot")

val df_ard_scid = df_ard_scid_period.select("dt", "scid_albumid", "hot")
                                    .union(df_ard_day_all_scid.select("dt", "scid_albumid", "hot"))
df_ard_scid.persist()
df_ard_scid.createOrReplaceTempView("table_ard_scid")


//save
val sql_ard_filter_scid = s"INSERT OVERWRITE TABLE temp.jimmy_scid_dt_hot_ard_zhudong PARTITION(cdt='$date_end') select dt, scid_albumid, hot from table_ard_scid"
spark.sql(sql_ard_filter_scid)
//consier delete save=====================================================

//unpersist to release memory
df_ard_day_scid_temp.unpersist()
df_ard_day_scid_unique.unpersist()
df_ard_day_all.unpersist()

//above sql is useful                                                                                                                
// scala> df_ard_day_scid_unique.count()
// res8: Long = 1655190      
// scala> df_ard_scid.select("scid_albumid").distinct.count()
// res10: Long = 1655190                                                                                                                   
 
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
df_period.persist()
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
df_cal_save.createOrReplaceTempView("cal_savetable")
val sql_burst_hot_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_burst_hot_score PARTITION(cdt='$date_end') select scid_albumid, burst, hot from cal_savetable"
spark.sql(sql_burst_hot_save)
//consier delete save=====================================================
//release to gain more memory
df_period.unpersist()
df_ard_scid.unpersist()
period_filter_temp.unpersist()

//standard the data, (data - mean)/std
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

//to up the minmum of hot-standard to ensure the sum operation below won't trouble in summing negative value
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
// val df_scid = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive" + $"positive2" + lit(1)).otherwise($"positive" + $"positive2" + lit(1)))
// we consier not to add lit(1) as bias
val df_scid = df_temp.withColumn("result_temp", when($"positive" > lit(2), lit(hot_coefficient) * $"positive" + $"positive2").otherwise($"positive" + $"positive2"))
                     .select($"scid_albumid", bround($"result_temp", 3) as "hot")  

//save values:hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min
var save_value_buf = new ListBuffer[(Double, Double, Double, Double, Double, Double)]()
save_value_buf += ((hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min))
val df_save_value = save_value_buf.toList
                                  .toDF("hot_mean", "hot_std", "burst_mean", "burst_std", "hot_coefficient", "hot_min")

df_save_value.createOrReplaceTempView("savevalue_table")
// df_save_value.persist()
val sql_save_value = s"INSERT OVERWRITE TABLE temp.jimmy_dt_save_value PARTITION(cdt='$date_end') select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from savevalue_table"
spark.sql(sql_save_value)

df_scid.persist()
df_scid.createOrReplaceTempView("table_scid")
val sql_scid_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_score PARTITION(cdt='$date_end') select scid_albumid, hot from table_scid"
spark.sql(sql_scid_save)
//unpersist to gain memory


//6)using df_scid to extract name from common.st_k_mixsong_part
//select singername, songname, albumname
val sql_sn="""
select a.mixsongid, a.choric_singer, a.songname, a.albumname, b.hot 
from common.st_k_mixsong_part a 
inner join 
table_scid b 
where a.dt = """+s"""'$date_end'"""+""" and a.mixsongid=b.scid_albumid
"""
val df_sn = spark.sql(sql_sn)
//filter and format name
val df_sn_sep = df_sn.withColumn("song", regexp_replace($"songname", "[ ]*\\([^\\(\\)]*\\)$", ""))
                     .withColumn("kind", regexp_extract($"songname", "\\(([^\\(\\)]*)\\)$", 1))
                     .withColumnRenamed("choric_singer", "singer") //to eliminate the space effect

df_sn_sep.persist()

df_sn_sep.createOrReplaceTempView("table_sep")
val sql_sn_sep_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_sn_sep PARTITION(cdt='$date_end') select mixsongid, albumname, song, kind, singer, hot from table_sep"
//remember that we should select the item in the hive item order!!!!!!!!!!!!!!!
spark.sql(sql_sn_sep_save)

//read
val sql_sn= s"select mixsongid, albumname, song, kind, singer, hot from temp.jimmy_dt_sn_sep where cdt='$date_end'"
val df_sn_sep = spark.sql(sql_sn)
df_sn_sep.persist()

////7)read remark and generate remark related dataframe
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// read remark
val sql_remark = "select scid_albumid, remark, hot, type, rel_album_audio_id from temp.jimmy_dt_hot_remark where cdt = "+s"'$date_end'"
val df_remark = spark.sql(sql_remark)
df_remark.persist()

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

// val window_cover_song = Window.partitionBy("cover_song")
// val window_song = Window.partitionBy("song")
// val df_ref_diff = df_remark_ref_final.filter($"song" =!= $"cover_song")
//                                        .withColumn("cover_value", sum($"cover_hot").over(window_cover_song))
//                                        .withColumn("value", max($"hot").over(window_song)) //change avg to max, cause it will exists many-many relation between song and hot, some hots are so small, it will lower the sum value, eg:凉凉
//                                        .withColumn("term",concat($"cover_song", lit(" 原唱")))
//                                        .withColumn("penality", $"value"*penality)
//                                        .groupBy("term").agg(max("penality") as "result")
//                                        .select("term", "result")
//                                        .withColumn("alias", lit("")) 

// val df_ref_eql = df_remark_ref_final.filter($"song" === $"cover_song")
//                                       .withColumn("cover_value", sum($"cover_hot").over(window_cover_song))
//                                       .withColumn("value", max($"hot").over(window_song)) //change avg to max, cause it will exists many-many relation between song and hot, some hots are so small, it will lower the sum value, eg:凉凉
//                                       .withColumn("term",concat($"cover_song", lit(" 原唱")))
//                                       .withColumn("penality", when($"cover_value" > $"value", $"value"*penality as "result").otherwise($"value"*penality*penality as "result"))
//                                       .groupBy("term").agg(max("penality") as "result")
//                                       .select("term", "result")
//                                       .withColumn("alias", lit(""))

@transient val window_cover_song = Window.partitionBy("cover_song")
@transient val window_song = Window.partitionBy("song")
val df_ref_diff_temp = df_remark_ref_final.filter($"song" =!= $"cover_song")
val df_ref_eql_temp = df_remark_ref_final.filter($"song" === $"cover_song")
@transient val diff_cover_value = sum(df_ref_diff_temp("cover_hot")).over(window_cover_song)
@transient val diff_value = max(df_ref_diff_temp("hot")).over(window_song)
//change avg to max, cause it will exists many-many relation between song and hot, some hots are so small, it will lower the sum value, eg:凉凉
@transient val eql_cover_value = sum(df_ref_eql_temp("cover_hot")).over(window_cover_song)
@transient val eql_value = max(df_ref_eql_temp("hot")).over(window_song)

val df_ref_diff = df_ref_diff_temp.withColumn("cover_value", diff_cover_value)
                                  .withColumn("value", diff_value)
                                  .withColumn("term",concat($"cover_song", lit(" 原唱")))
                                  .withColumn("penality", $"value"*penality)
                                  .groupBy("term").agg(max("penality") as "result")
                                  .select("term", "result")

val df_ref_eql = df_ref_eql_temp.withColumn("cover_value", eql_cover_value)
                                .withColumn("value", eql_value)
                                .withColumn("term",concat($"cover_song", lit(" 原唱")))
                                .withColumn("penality", when($"cover_value" > $"value", $"value"*penality as "result").otherwise($"value"*penality*penality*penality as "result"))
                                .groupBy("term").agg(max("penality") as "result")
                                .select("term", "result")

val df_ref = df_ref_eql.union(df_ref_diff)
                       .groupBy("term").agg(max("result") as "result")
                       .withColumn("alias", lit(""))
                       .select("term", "alias","result")//change the order otherwise it will change the column result from double to string

//generate index

//9) union all related dataframe to df_final(created by song self)
val df_term = df_singersong.union(df_songsinger)
                           .union(df_singer)
                           .union(df_song)
                           .union(df_songkind)
                           .union(df_album)
                           .union(df_notion)
                           .union(df_notioncategory)
                           .union(df_language)
                           .union(df_translate)
                           .groupBy("term", "alias").agg(max("result") as "result")
//change some alias
//change some alias
df_ref.persist()
df_ref.count()

//we wait df_ref then run this
df_term.persist()
df_term.count()
//change some alias

val df_final = df_term.select("term", "alias","result")
                      .union(df_ref)
                      .groupBy("term", "alias").agg(max("result") as "result")
                      .select($"term", bround($"result", 3) as "result", $"alias")  

df_final.persist()

df_final.createOrReplaceTempView("term_savetable")
val sql_term_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result PARTITION(cdt='$date_end') select term, result, alias from term_savetable"
spark.sql(sql_term_save)

//check the result
// df_final.filter(col("term").startsWith("起风了")).sort($"result".desc).show()

//10)manual input
//merge all and groupby to df_kw_manual with kw,hot
//read
val sql_ard_kw_read= s"select kw, hot from temp.jimmy_kw_hot_ard_zhudong where cdt = '$date_end'"
val df_kw_all = spark.sql(sql_ard_kw_read)
df_kw_all.persist()   

//standard the data, (data - mean)/std
//use save value from yesterday
val sql_read_value = s"select hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min from temp.jimmy_dt_save_value where cdt='$date_end'"
val df_save_value = spark.sql(sql_read_value)
val (hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min) = (df_save_value.first.getDouble(0), df_save_value.first.getDouble(1), df_save_value.first.getDouble(2), df_save_value.first.getDouble(3), df_save_value.first.getDouble(4), df_save_value.first.getDouble(5))

//use saved values:hot_mean, hot_std, burst_mean, burst_std, hot_coefficient, hot_min
val df_kw_cal = df_kw_all.withColumn("hot_standard", ($"hot"-hot_mean)/hot_std)

//we only use the positive part of hot_standard
val df_kw_manual = df_kw_cal.withColumn("positive2", $"hot_standard" + lit(hot_min))
                            .filter($"positive2" > 0)
                            .select($"kw", bround($"positive2", 3) as "hot")

df_kw_manual.persist()
df_kw_manual.createOrReplaceTempView("table_kw_manual")
val sql_kw_save = s"INSERT OVERWRITE TABLE temp.jimmy_dt_kw_hot_score PARTITION(cdt='$date_end') select kw, hot from table_kw_manual"
spark.sql(sql_kw_save)


//11) merge kw and df_final to df_mix with term, result, alias
val df_kw_manual_strip = df_kw_manual.withColumn("origin2", mystrip($"kw"))
                                     .groupBy("origin2").agg(max("hot") as "hot")
val df_final_strip = df_final.withColumn("origin", mystrip($"term"))

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

val df_final_exist = df_final.join(df_anti, df_final("term") === df_anti("term_out"), "leftanti")
                             .select("term", "result", "alias")

val df_kw_manual_exist = df_anti.groupBy("origin2").agg(max("hot") as "result")
                                .withColumnRenamed("origin2", "term")
                                .withColumn("alias", myalias($"term"))
                                .select("term", "result", "alias")

val df_mix = df_final_exist.union(df_kw_manual_exist)
                           .union(df_kw_manual_unique)
                           .groupBy("term", "alias").agg(max("result") as "result")
df_mix.persist()
df_mix.count()

//12) add the tag of variety to df_search with term, result, alias

val sql_kw_variety_album = s"""
select a.singername, a.albumid 
from common.k_singer_album_part a 
LEFT SEMI JOIN 
(select singerid 
from common.k_singer_part 
where dt='$date_end' and sextype = '12' 
group by singerid)b 
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
val sql_scid_read="select scid_albumid, hot from temp.jimmy_dt_hot_score where cdt = "+s"'$date_end'"
val df_scid = spark.sql(sql_scid_read)
df_scid.persist()
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
df_search.count()
//check the result
// df_search.filter(col("term").startsWith("起风了")).sort($"result".desc).show()


//13) read from specified tag


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
// [Stage df_scid_hot.count()
// res3: Long = 1409489                                                            
// scala> df_scid.count()
// res4: Long = 1410153 

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
df_scid_tag_hot.persist()
df_scid_tag_hot.count()
//save
df_scid_tag_hot.createOrReplaceTempView("table_scid_tag_hot")
val sql_scid_tag_hot_save = s"INSERT OVERWRITE TABLE temp.jimmy_scid_tag_hot PARTITION(cdt='$date_end') select scid, tag, hot from table_scid_tag_hot"
spark.sql(sql_scid_tag_hot_save)
// scala> df_scid_hot.count()
// res39: Long = 1311781                                                           
// scala> df_scid_tag_hot.count()
// res40: Long = 1178331
val sql_tag_keyword = """
select
        tag,
        kw
from temp.jimmy_tag
"""

val df_tag_keyword = spark.sql(sql_tag_keyword)
df_tag_keyword.persist()
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
val sql_search = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_mix PARTITION(cdt='$date_end') select term, result, alias from search_savetable"
spark.sql(sql_search)
//check result
// df_triple.filter(col("term").startsWith("起风了")).sort($"result".desc).show()

//read
// val sql_triple= s"select term, result, alias from temp.jimmy_hot_result_mix where cdt='$date_end'"
// val df_triple = spark.sql(sql_triple)
// df_triple.persist()

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
val sql_result_num = s"INSERT OVERWRITE TABLE temp.jimmy_dt_hot_result_num_mix PARTITION(cdt='$date_end') select * from num_search_savetable"
spark.sql(sql_result_num)