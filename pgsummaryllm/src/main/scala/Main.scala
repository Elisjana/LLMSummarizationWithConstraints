package pgsummaryllm

import org.apache.spark.sql.{SparkSession, DataFrame, Row, Column}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.api.java.UDF2

import com.typesafe.config.{Config, ConfigFactory}
import java.nio.file.{Files, Paths}
import java.io.File

object Main {

  // ------------------------------------------------------------
  // Find conf/application.conf by walking up parent dirs
  // ------------------------------------------------------------
  def findConf(startDir: String): File = {
    var p = Paths.get(startDir).toAbsolutePath.normalize()
    while (p != null && !Files.exists(p.resolve("conf").resolve("application.conf"))) {
      p = p.getParent
    }
    if (p == null) throw new RuntimeException("Could not find conf/application.conf in any parent directory")
    p.resolve("conf").resolve("application.conf").toFile
  }

  // ------------------------------------------------------------
  // Helper: normalize schema node name (Type vs Label)
  // ------------------------------------------------------------
  def normalizeSchemaNodeName(
      s: String,
      typeToLabel: Map[String, String],
      labelSet: Set[String]
  ): String = {
    if (s == null) return null
    if (typeToLabel.contains(s)) typeToLabel(s) // PersonType -> Person
    else if (labelSet.contains(s)) s            // Person stays Person
    else if (s.endsWith("Type") && labelSet.contains(s.stripSuffix("Type"))) s.stripSuffix("Type")
    else s
  }

  def main(args: Array[String]): Unit = {

    implicit val spark: SparkSession =
      SparkSession.builder()
        .appName("LLM-Summarizer-Linker")
        .master("local[*]")
        .getOrCreate()

    import spark.implicits._

    spark.sparkContext.setLogLevel("WARN")

    val confFile = findConf(System.getProperty("user.dir"))
    println(s"[info] Loading config from: ${confFile.getAbsolutePath}, exists=${confFile.exists()}")

    val conf: Config = ConfigFactory.parseFile(confFile).resolve()

    // ---------------------------
    // INPUTS
    // ---------------------------
    val schemaPath      = conf.getString("paths.schema")
    val constraintsPath = conf.getString("paths.jsonConstraints")

    println(s"[info] schemaPath=$schemaPath")
    println(s"[info] constraintsPath=$constraintsPath")

    // ---------------------------
    // OUTPUTS
    // ---------------------------
    val outLinked = "out/linked_constraints"
    val outErrors = "out/link_errors"

    // ---------------------------
    // PARSE SCHEMA
    // ---------------------------
    val ddlText = spark.read.textFile(schemaPath).collect().mkString("\n")
    val catalog = Linker.parseSchemaDDL(ddlText)

    // DEBUG: show some schema rel endpoints
    println("[info] === schema rels (first 10) ===")
    catalog.rels.take(10).foreach { case (k, v) =>
      println(s"[info] $k : ${v.srcLabel} -> ${v.dstLabel}")
    }

    // ------------------------------------------------------------
    // Build schema maps for normalization
    // ------------------------------------------------------------
    val schemaTypeToLabel: Map[String, String] =
      catalog.nodes.map { case (typeName, nt) => typeName -> nt.label }

    val schemaLabels: Set[String] =
      catalog.nodes.values.map(_.label).toSet

    val bcTypeToLabel = spark.sparkContext.broadcast(schemaTypeToLabel)
    val bcLabels      = spark.sparkContext.broadcast(schemaLabels)

    // UDF: normalize constraint node label into schema label space
    val normalizeNodeLabelUdf: UserDefinedFunction =
      udf { s: String =>
        normalizeSchemaNodeName(s, bcTypeToLabel.value, bcLabels.value)
      }

    // ------------------------------------------------------------
    // 1) Rel aliases to match schema rel labels
    // (EDIT this map if your schema uses different names)
    // ------------------------------------------------------------
    val relAlias: Map[String, String] = Map(
      "WORK_AT"  -> "WORKS_AT",
      "STUDY_AT" -> "STUDIES_AT"
    )
    val bcRelAlias = spark.sparkContext.broadcast(relAlias)

    // ------------------------------------------------------------
    // 2) Build schema endpoints map (normalized to label space)
    // IMPORTANT: normalize src/dst too (they may be Type names)
    // ------------------------------------------------------------
    val relEndpoints: Map[String, (String, String)] =
      catalog.rels.map { case (_, rt) =>
        val relLabel = rt.label
        val expSrc   = normalizeSchemaNodeName(rt.srcLabel, schemaTypeToLabel, schemaLabels)
        val expDst   = normalizeSchemaNodeName(rt.dstLabel, schemaTypeToLabel, schemaLabels)
        relLabel -> (expSrc, expDst)
      }
    val bcRelEndpoints = spark.sparkContext.broadcast(relEndpoints)

    // ------------------------------------------------------------
    // Output type for fixed edges: array<struct<e,label,src,dst,props>>
    // ------------------------------------------------------------
    val edgeOutType: DataType =
      ArrayType(
        new StructType()
          .add("e", "string", nullable = true)
          .add("label", "string", nullable = true)
          .add("src", "string", nullable = true)
          .add("dst", "string", nullable = true)
          .add("props", ArrayType(StringType), nullable = true),
        containsNull = true
      )

    // ------------------------------------------------------------
    // UDF: normalize rel label + swap src/dst if reversed vs schema
    // ------------------------------------------------------------
    val fixEdgesUdf: UserDefinedFunction =
      udf(
        new UDF2[Seq[Row], Seq[Row], Seq[Row]] {
          override def call(edges: Seq[Row], nodes: Seq[Row]): Seq[Row] = {

            val var2label: Map[String, String] =
              Option(nodes).getOrElse(Seq.empty).flatMap { n =>
                val v = Option(n.getAs[String]("v"))
                val l = Option(n.getAs[String]("label"))
                if (v.isDefined && l.isDefined) Some(v.get -> l.get) else None
              }.toMap

            val alias = bcRelAlias.value
            val ep    = bcRelEndpoints.value

            Option(edges).getOrElse(Seq.empty).map { e =>
              val eVar = e.getAs[String]("e")
              val rawL = e.getAs[String]("label")
              val lbl  = alias.getOrElse(rawL, rawL)

              val srcV = e.getAs[String]("src")
              val dstV = e.getAs[String]("dst")

              val hasProps = e.schema.fieldNames.contains("props")
              val props: Seq[String] =
                if (hasProps) Option(e.getAs[Seq[String]]("props")).getOrElse(Seq.empty[String])
                else Seq.empty[String]

              val srcLabel = var2label.getOrElse(srcV, "")
              val dstLabel = var2label.getOrElse(dstV, "")

              ep.get(lbl) match {
                case Some((expSrc, expDst)) =>
                  val reversed = (srcLabel == expDst && dstLabel == expSrc)
                  if (reversed) Row(eVar, lbl, dstV, srcV, props)
                  else          Row(eVar, lbl, srcV, dstV, props)
                case None =>
                  Row(eVar, lbl, srcV, dstV, props)
              }
            }
          }
        },
        edgeOutType
      )

    // ------------------------------------------------------------
    // Read constraints JSON
    // ------------------------------------------------------------
    val raw = spark.read.json(constraintsPath)

    val rawNodeElemType = StructType(Seq(
      StructField("Node label", StringType, true),
      StructField("var", StringType, true)
    ))

    val rawEdgeElemType = StructType(Seq(
      StructField("Edge label", StringType, true),
      StructField("src", StringType, true),
      StructField("dst", StringType, true),
      StructField("var", StringType, true)
    ))

    def emptyRawNodes: Column = expr("array()").cast(ArrayType(rawNodeElemType))
    def emptyRawEdges: Column = expr("array()").cast(ArrayType(rawEdgeElemType))

    def rawNodesCol(side: String): Column =
      when(col(s"$side.nodes").isNull, emptyRawNodes)
        .otherwise(col(s"$side.nodes").cast(ArrayType(rawNodeElemType)))

    def rawEdgesCol(side: String): Column =
      when(col(s"$side.edges").isNull, emptyRawEdges)
        .otherwise(col(s"$side.edges").cast(ArrayType(rawEdgeElemType)))

    // Normalize nodes to: struct(v,label,props)
    def normNodes(rawNodes: Column): Column =
      transform(rawNodes, n =>
        struct(
          n.getField("var").as("v"),
          normalizeNodeLabelUdf(n.getField("Node label")).as("label"),
          array().cast("array<string>").as("props")
        )
      )

    // Normalize edges to: struct(e,label,src,dst,props)
    def normEdges(rawEdges: Column): Column =
      transform(rawEdges, e =>
        struct(
          e.getField("var").as("e"),
          e.getField("Edge label").as("label"),
          e.getField("src").as("src"),
          e.getField("dst").as("dst"),
          array().cast("array<string>").as("props")
        )
      )

    // Build constraintsDF with consistent schema
    val constraintsDF =
      raw
        .withColumn("lhs", struct(
          normNodes(rawNodesCol("lhs")).as("nodes"),
          normEdges(rawEdgesCol("lhs")).as("edges")
        ))
        .withColumn("rhs", struct(
          normNodes(rawNodesCol("rhs")).as("nodes"),
          normEdges(rawEdgesCol("rhs")).as("edges")
        ))
        .select(
          col("id").as("cid"),
          col("type").as("ctype"),
          col("lhs"),
          col("rhs")
        )

    // Quick sanity checks
    constraintsDF.selectExpr("cid","ctype","size(lhs.edges) as lhsE","size(rhs.edges) as rhsE").show(10,false)

    // ------------------------------------------------------------
    // Fix edges using schema endpoints (alias + reverse if needed)
    // ------------------------------------------------------------
    val constraintsFixed =
      constraintsDF
        .withColumn("lhs", struct(
          col("lhs.nodes").as("nodes"),
          fixEdgesUdf(col("lhs.edges"), col("lhs.nodes")).as("edges")
        ))
        .withColumn("rhs", struct(
          col("rhs.nodes").as("nodes"),
          fixEdgesUdf(col("rhs.edges"), col("rhs.nodes")).as("edges")
        ))

    // ============================================================
    // ✅ Correct linking (replaces Linker.linkConstraints)
    // ============================================================

    // Canonical schema edges DF: (edgeLabel, srcLabel, dstLabel)
    val schemaCanon: DataFrame =
      catalog.rels.values.toSeq
        .map { rt =>
          val relLabel = rt.label
          val srcLab   = normalizeSchemaNodeName(rt.srcLabel, schemaTypeToLabel, schemaLabels)
          val dstLab   = normalizeSchemaNodeName(rt.dstLabel, schemaTypeToLabel, schemaLabels)
          (relLabel, srcLab, dstLab)
        }
        .toDF("edgeLabel", "srcLabel", "dstLabel")
        .select(
          upper(trim(col("edgeLabel"))).as("edgeLabel"),
          upper(trim(col("srcLabel"))).as("srcLabel"),
          upper(trim(col("dstLabel"))).as("dstLabel")
        )
        .dropDuplicates()

    def computeSideLink(side: String, df: DataFrame): (DataFrame, DataFrame) = {
      val nodesCol = col(s"$side.nodes")
      val edgesCol = col(s"$side.edges")

      val varMapExploded =
        df.select(col("cid"), explode_outer(nodesCol).as("n"))
          .select(
            col("cid"),
            col("n.v").as("var"),
            upper(trim(col("n.label"))).as("nlabel")
          )

      val edgesExploded =
        df.select(col("cid"), explode_outer(edgesCol).as("e"))
          .select(
            col("cid"),
            upper(trim(col("e.label"))).as("edgeLabel"),
            col("e.src").as("srcVar"),
            col("e.dst").as("dstVar")
          )

      val srcMap = varMapExploded.select(col("cid"), col("var").as("srcVar"), col("nlabel").as("srcLabel"))
      val dstMap = varMapExploded.select(col("cid"), col("var").as("dstVar"), col("nlabel").as("dstLabel"))

      val edgesWithLabels =
        edgesExploded
          .join(srcMap, Seq("cid","srcVar"), "left")
          .join(dstMap, Seq("cid","dstVar"), "left")

      val edgesNorm =
        edgesWithLabels.select(
          col("cid"),
          col("edgeLabel"),
          upper(trim(col("srcLabel"))).as("srcLabel"),
          upper(trim(col("dstLabel"))).as("dstLabel")
        )

      val matched =
        edgesNorm.join(schemaCanon, Seq("edgeLabel","srcLabel","dstLabel"), "left_semi")

      val okDF =
        edgesNorm.groupBy("cid").agg(count(lit(1)).as("nEdges"))
          .join(matched.groupBy("cid").agg(count(lit(1)).as("nMatched")), Seq("cid"), "left")
          .withColumn("nMatched", coalesce(col("nMatched"), lit(0)))
          .withColumn(s"${side}_linked_ok", col("nMatched") === col("nEdges"))
          .select("cid", s"${side}_linked_ok")

      val errDF =
        edgesNorm
          .join(schemaCanon, Seq("edgeLabel","srcLabel","dstLabel"), "left_anti")
          .withColumn("side", lit(side))
          .select("cid","side","edgeLabel","srcLabel","dstLabel")

      (okDF, errDF)
    }

    val (lhsOK, lhsErr) = computeSideLink("lhs", constraintsFixed)
    val (rhsOK, rhsErr) = computeSideLink("rhs", constraintsFixed)

    val linked =
      constraintsFixed
        .join(lhsOK, Seq("cid"), "left")
        .join(rhsOK, Seq("cid"), "left")
        .withColumn("lhs_linked_ok", coalesce(col("lhs_linked_ok"), lit(true)))
        .withColumn("rhs_linked_ok", coalesce(col("rhs_linked_ok"), lit(true)))
        .withColumn("linked_ok", col("lhs_linked_ok") && col("rhs_linked_ok"))

    val errors = lhsErr.unionByName(rhsErr)

    // ------------------------------------------------------------
    // ✅ Attach link_errors into the linked DF (so ConstraintSummarizer works)
    // ------------------------------------------------------------
    val errAgg =
      errors
        .groupBy("cid")
        .agg(
          collect_list(
            concat_ws(" ",
              col("side"),
              col("edgeLabel"),
              lit("("), col("srcLabel"), lit("->"), col("dstLabel"), lit(")")
            )
          ).as("link_errors")
        )

    val linkedWithErrors =
      linked
        .join(errAgg, Seq("cid"), "left")
        .withColumn("link_errors", coalesce(col("link_errors"), array()))

    linkedWithErrors.select("cid","ctype","lhs_linked_ok","rhs_linked_ok","linked_ok").show(50,false)
    errors.show(50,false)

    println("[debug] outLinked schema to be written:")
    linkedWithErrors.printSchema()

    // Write outputs
    linkedWithErrors.write.mode("overwrite").parquet(outLinked)
    linkedWithErrors.write.mode("overwrite").parquet(outErrors)


    println("[info] Linker finished successfully")
    println(s"[info] outLinked=$outLinked")
    println(s"[info] outErrors=$outErrors")

    // ------------------------------------------------------------
    // Summaries
    // ------------------------------------------------------------
    val linkedOutput = spark.read.parquet(outLinked)

    // Compatibility columns for older summarizers (DO NOT flatten lhs/rhs!)
    val linkedForSummaries =
      linkedOutput
        .withColumn("lhs_nodes_linked", col("lhs.nodes"))
        .withColumn("lhs_edges_linked", col("lhs.edges"))
        .withColumn("rhs_nodes_linked", col("rhs.nodes"))
        .withColumn("rhs_edges_linked", col("rhs.edges"))
        .withColumn("link_errors", coalesce(col("link_errors"), array()))

    val nodeTypeSumm = SchemaSummarizer.summarizeNodeTypes(catalog, linkedForSummaries)
    val relTypeSumm  = SchemaSummarizer.summarizeRelTypes(catalog, linkedForSummaries)
    val consSumm     = ConstraintSummarizer.summarizeConstraints(linkedForSummaries)

    nodeTypeSumm.write.mode("overwrite").json(conf.getString("paths.outNodeTypeSummaries"))
    relTypeSumm.write.mode("overwrite").json(conf.getString("paths.outRelTypeSummaries"))
    consSumm.write.mode("overwrite").json(conf.getString("paths.outConstraintSummaries"))

    println("[info] ✅ Summaries written for LLM ingestion")


    
    // --------------------------------------------------
    // Run validator
    // --------------------------------------------------
      val cfg = GemmaInputBuilder.Config(
        schemaPath           = "/mnt/c/Projects/PhD/LLMSummarizationWithConstraints/schema.json",
        constraintsJsonlPath = "/mnt/c/Projects/PhD/LLMSummarizationWithConstraints/JsonOutput/ldbc/v1/constraints.jsonl",

        // NEW CONFIG FIELDS
        maxConstraintsTotal      = 5000,  // set lower if you want, e.g. 200
        maxConstraintsPerType    = 5000,
        maxConstraintsPerUnknown = 5000,

        includePrettySchemaJson  = false, // keep false (recommended)
        maxSchemaPrettyChars     = 6000,  // only used if includePrettySchemaJson=true

        maxSideNodes             = 12,
        maxSideEdges             = 18,
        maxLinkErrors            = 8,

        normalizeEdgeLabels      = true,  // maps WORK_AT->WORKS_AT, STUDY_AT->STUDIES_AT

        outJsonlPath             = "/mnt/c/Projects/PhD/LLMSummarizationWithConstraints/summaryOutput/gemma_inputs_per_constraint.jsonl"
      )

      GemmaInputBuilder.build(spark, cfg)

    spark.stop()
  }
}