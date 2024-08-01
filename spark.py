from pyspark.sql import SparkSession, functions as sf, Window as w
import os

def run():
  os.environ["PYARROW_IGNORE_TIMEZONE"] = "1" 

  spark = SparkSession.builder.getOrCreate()
  spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
  crime_file = '/content/drive/MyDrive/otus/spark/crime.csv'
  offense_codes_file = '/content/drive/MyDrive/otus/spark/offense_codes.csv'
  path_to_out_file = "/content/my_data_mart.parquet"

  df = spark.read.csv(crime_file, header=True, inferSchema=True)
  df_dist = df.dropDuplicates()

  w_district  = w.partitionBy("DISTRICT")
  w_district_y_m  = w.partitionBy(["DISTRICT", "YEAR", "MONTH"])
  w_district_m  = w.partitionBy(["DISTRICT", "MONTH"])
  w_rank_frequent_crime_types = w.partitionBy('DISTRICT').orderBy('crimes_total')

  df_dist_frequent_crime_types = df_dist \
        .filter(df_dist.DISTRICT != 'NULL') \
        .withColumn("crimes_total", sf.count(df_dist.DISTRICT).over(w_district)) \
        .withColumn('crime_type', sf.split(df_dist.OFFENSE_DESCRIPTION, ' - ', 2)[0]) \
        .withColumn("rank_frequent_crime_types", sf.row_number().over(w_rank_frequent_crime_types)) \
        .where(sf.col('rank_frequent_crime_types') <=3) \
        .groupBy('DISTRICT') \
        .agg(sf.concat_ws(',' ,sf.collect_list('crime_type')).alias('frequent_crime_types')) \
        .select('DISTRICT', "frequent_crime_types")

  data_mart = df_dist \
        .join(df_dist_frequent_crime_types, 'DISTRICT') \
        .filter(df_dist.DISTRICT != 'NULL') \
        .withColumn("crimes_total", sf.count(df_dist.DISTRICT).over(w_district)) \
        .withColumn("crimes_month", sf.count(df_dist.DISTRICT).over(w_district_y_m)) \
        .withColumn("crimes_monthly", sf.percentile_approx("crimes_month", 0.5, 10000).over(w.partitionBy('DISTRICT').orderBy('crimes_total'))) \
        .withColumn('crime_type', sf.split(df_dist.OFFENSE_DESCRIPTION, ' - ', 2)[0]) \
        .withColumn("lat", sf.mean(df_dist.Lat).over(w_district)) \
        .withColumn("lng", sf.mean(df_dist.Long).over(w_district)) \
        .select('DISTRICT', 'YEAR', 'MONTH', 'crimes_total', 'crimes_monthly', 'frequent_crime_types', 'crime_type', 'lat', 'lng') \
        .orderBy(['DISTRICT', 'YEAR', 'MONTH'])
  
  data_mart.write.parquet(path_to_out_file)

  spark.stop()

if __name__ == "__main__":
  run()
