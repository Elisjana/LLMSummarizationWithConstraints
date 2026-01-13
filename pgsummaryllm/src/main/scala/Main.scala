import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import java.nio.file.{Files, Paths}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.File


object WordsCount {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("WordsCount")
      .master("local[*]")
      .config("spark.ui.showConsoleProgress", "false")
      .getOrCreate()

      spark.sparkContext.setLogLevel("WARN")

      import spark.implicits._ 
      val sc = spark.sparkContext


      //############################################################## INPUT / OUTPUT PATHS ####################################################################
      val inputDir  = "/mnt/c/Projects/HY562-Assignment-1/loadFiles"
      val inputPath = s"file://$inputDir/*.txt"
      val outputPath = "file:///mnt/c/Projects/HY562-Assignment-1/output/WordsCount_csv"

      if (!Files.exists(Paths.get(inputDir))) {
        System.err.println(s"Input directory does not exist: $inputDir")
        spark.stop()
        System.exit(1)
      }


    //############################################################## CHECK IF DIRECTORY AND .TXT FILES EXIST ####################################################################
      val dir = new File(inputDir)
      if (dir.exists && dir.isDirectory) {
        val txtFiles = Option(dir.listFiles).getOrElse(Array.empty)
          .filter(f => f.isFile && f.getName.endsWith(".txt"))

        if (txtFiles.nonEmpty) {
          val sc = spark.sparkContext
          val textRDD: RDD[String] = sc.textFile(inputPath)
          println(s"Loaded ${txtFiles.length} text files.")


          
          //############################################################## DELETE OLD DIRECTORY ####################################################################
          val fs = FileSystem.get(sc.hadoopConfiguration)
          val outPath = new Path(outputPath)
          if (fs.exists(outPath)) {
            println(s"Output path exists. Deleting: $outputPath")
            fs.delete(outPath, true)
          } else {
            println("Output path does not exist.")
          }

          
          //############################################################## READ, TOKENIZE, & COUNT WORDS ####################################################################
          val tokens: RDD[String] =
            sc.textFile(inputPath)
              .flatMap(_.split("\\W+"))
              .filter(_.nonEmpty)
              .map(_.toLowerCase)

          val wordCounts: RDD[(String, Int)] =
            tokens
              .map(word => (word, 1))
              .reduceByKey(_ + _)
              .sortBy({ case (word, count) => (-count, word) })


          
          //############################################################## PRINT N=50 WORDS JUST FOR CHECK IN CONSOLE ####################################################################
          //wordCounts.take(50).foreach { case (word, count) =>
          //  println(s"$word $count")
          //}

          
          //############################################################## WRITE RESULTS AS CSV ####################################################################
          val df = wordCounts.toDF("word", "count")
          df.coalesce(1)
            .write
            .mode("overwrite")
            .option("header", "true")
            .csv(outputPath)


      }else {
          println("No .txt files found — skipping load.")
        }

    } else {
      println(s"Directory does not exist: $inputDir")
    }

      spark.stop()
    }
}