package dataparserModule

import com.typesafe.config.{Config, ConfigFactory}
import java.io.File
import java.nio.file.Paths


object Main {

    def main(args: Array[String]): Unit = {

        val conf: Config = {
            val baseDir = System.getProperty("user.dir")
            val file    = new File(baseDir + "/conf/application.conf")
            println(s"Loading config from: ${file.getAbsolutePath}, exists=${file.exists()}")
            ConfigFactory.parseFile(file).resolve()
        }

        val inputRootDir = conf.getString("constraintSerializer.inputRoot")   
        val defaultVer   = conf.getString("constraintSerializer.defaultVersion") 
        val outputRootDir= conf.getString("constraintSerializer.outputRoot")   

        val inputRoot = Paths.get(inputRootDir)
        val version   = if (args.nonEmpty) args(0) else defaultVer
        val outRoot   = Paths.get(outputRootDir)


        ConstraintSerializer.processAll(inputRoot, version, outRoot)
    }
}