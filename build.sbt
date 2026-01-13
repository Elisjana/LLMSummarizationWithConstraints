ThisBuild / scalaVersion := "2.12.18"             
ThisBuild / version      := "0.1.0-SNAPSHOT"

lazy val root = (project in file("."))
  .settings(
    name := "LLMSummarizationWithConstraints",
    libraryDependencies ++= Seq(
        "org.apache.spark" %% "spark-core"   % "3.4.1",   
        "org.apache.spark" %% "spark-sql"    % "3.4.1",
        "org.apache.spark" %% "spark-mllib"  % "3.4.1",
        "org.scala-lang"    % "scala-reflect" % scalaVersion.value
    ),

    javaHome := Some(file("/usr/lib/jvm/java-11-openjdk-amd64")),
   
    Compile / run / fork := true,
    javaOptions in run ++= Seq(
      "-Xms512m", 
      "-Xmx2g",
      "-Dspark.executor.memory=32G",
      "-Dspark.driver.memory=32G",
      "-Dspark.driver.total.executor.cores=96",
      "-Dspark.executor.cores=8",
      "-Dspark.driver.cores=3",
      "-Dspark.executor.instances=10",
      "-Dspark.yarn.executor.memoryOverhead=8G",
      "-Dspark.driver.maxResultSize=16G"
    ) 
  )