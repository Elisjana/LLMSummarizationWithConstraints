import sbt._
import sbt.Keys._
import sbt.librarymanagement.Resolver

val scalaV       = "2.12.18"
val sparkV       = "3.4.1"
val circeV       = "0.14.7"
val scoptV       = "4.1.0"
val neo4jDriverV = "4.4.14"

ThisBuild / scalaVersion := scalaV
ThisBuild / organization := "pgsummaryllm"
ThisBuild / version      := "0.1.0-SNAPSHOT"

ThisBuild / resolvers ++= Seq(Resolver.mavenCentral)

lazy val commonSettings = Seq(
  scalacOptions ++= Seq(
    "-deprecation",
    "-feature",
    "-unchecked",
    "-encoding", "utf-8"
    // Tip: remove "-Xfatal-warnings" if it blocks you often
    // "-Xfatal-warnings"
  ),
  libraryDependencies += "com.typesafe" % "config" % "1.4.3",
  Test / fork := true,
  Test / parallelExecution := false
)

// ==================================================
// PROJECT: utils
// ==================================================
lazy val utils = (project in file("utils"))
  .settings(commonSettings)
  .settings(
    name := "utils",
    libraryDependencies ++= Seq(
      "io.circe" %% "circe-core"    % circeV,
      "io.circe" %% "circe-generic" % circeV,
      "io.circe" %% "circe-parser"  % circeV,
      "com.github.scopt" %% "scopt" % scoptV,
      "com.lihaoyi" %% "ujson"      % "3.2.0",
      "org.neo4j.driver" % "neo4j-java-driver" % neo4jDriverV,
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    )
  )

// ==================================================
// MODULE: dataparserModule
// ==================================================
lazy val dataparserModule = (project in file("dataparserModule"))
  .settings(commonSettings)
  .settings(
    name := "dataparserModule",
    libraryDependencies ++= Seq(
      "io.circe" %% "circe-core"    % circeV,
      "io.circe" %% "circe-generic" % circeV,
      "io.circe" %% "circe-parser"  % circeV,
      "com.github.scopt" %% "scopt" % scoptV,
      "com.lihaoyi" %% "ujson"      % "3.2.0",
      "org.neo4j.driver" % "neo4j-java-driver" % neo4jDriverV,
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    )
  )
  .dependsOn(utils)

// ==================================================
// MODULE: pgsummaryllm
// ==================================================
lazy val pgsummaryllm = (project in file("pgsummaryllm"))
  .settings(commonSettings)
  .settings(
    name := "pgsummaryllm",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkV % "provided",
      "org.apache.spark" %% "spark-sql"  % sparkV,
      "org.neo4j.driver" % "neo4j-java-driver" % neo4jDriverV,
      "io.circe" %% "circe-core"    % circeV,
      "io.circe" %% "circe-generic" % circeV,
      "io.circe" %% "circe-parser"  % circeV,
      "com.lihaoyi" %% "ujson"      % "3.2.0",
      "org.scalatest" %% "scalatest" % "3.2.19" % Test
    ),
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq("-Xms1G", "-Xmx4G")
  )
  .dependsOn(dataparserModule, utils)

// ==================================================
// ROOT aggregator (ONLY ONE root)
// ==================================================
lazy val root = (project in file("."))
  .settings(
    name := "LLMSummarizationWithConstraints",
    Compile / unmanagedResourceDirectories += baseDirectory.value / "conf",

    // If you want javaHome + spark options for running from root:
    javaHome := Some(file("/usr/lib/jvm/java-11-openjdk-amd64")),
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq(
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
  .aggregate(utils, dataparserModule, pgsummaryllm)