package pgsummaryllm

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.util.matching.Regex
import scala.util.Try
import org.apache.spark.sql.Row
import org.apache.spark.sql.Column


// File: src/main/scala/pgpartitioner/Linker.scala
// Step 1 — Linker (schema ↔ constraints)
// - Reads a property-graph schema DDL (CREATE NODE TYPE / CREATE REL TYPE style)
// - Reads constraints as JSONL
// - Validates & annotates constraints against the schema
// - Writes: linked_constraints (JSONL or Parquet) + a separate errors report
//
// Assumptions about constraints JSONL (adapt to your fields if needed):
// {
//   "cid":"c1",
//   "ctype":"gfd" | "ggd" | "pgkey",
//   "lhs": { "nodes":[{"v":"a","label":"Person","props":["id","firstName"]}], "edges":[{"e":"e1","label":"KNOWS","src":"a","dst":"b","props":[]}] },
//   "rhs": { "nodes":[...], "edges":[...] }
// }


object Linker {

  // ----------------------------
  // 1) Schema model
  // ----------------------------
  final case class NodeType(label: String, props: Set[String])
  final case class RelType(label: String, srcLabel: String, dstLabel: String, props: Set[String])

  final case class SchemaCatalog(
    nodes: Map[String, NodeType],
    rels: Map[String, RelType]
  )

  // ----------------------------
  // 2) Parse schema DDL
  // ----------------------------
  private val NodeBlock: Regex =
    """CREATE\s+NODE\s+TYPE\s*\(\s*\w+\s*:\s*([A-Za-z_]\w*)\s*\{([\s\S]*?)\}\s*\)\s*;""".r

  private val RelBlock: Regex =
    ("""CREATE\s+EDGE\s+TYPE\s*""" +
      """\(\s*:\s*([A-Za-z_]\w*(?:\s*\|\s*[A-Za-z_]\w*)*)\s*\)\s*""" +   // ( : A | B )  -> group(1)
      """-\s*\[\s*([A-Za-z_]\w*)\s*(?:\{([\s\S]*?)\})?\s*\]\s*""" +      // [ EDGE { props } ] -> group(2)=EDGE, group(3)=props (optional)
      """->\s*\(\s*:\s*([A-Za-z_]\w*(?:\s*\|\s*[A-Za-z_]\w*)*)\s*\)\s*;""").r // ( : C | D ) -> group(4)


  private def parseProps(block: String): Set[String] = {
    // matches: id String, OPTIONAL content String, etc.
    // keeps property names only
    val PropLine = """(?m)^\s*(OPTIONAL\s+)?([A-Za-z_]\w*)\s+[\w\[\]]+\s*,?\s*$""".r
    PropLine.findAllMatchIn(block).map(m => m.group(2)).toSet
  }

  def parseSchemaDDL(ddlText: String): SchemaCatalog = {
    val nodes = NodeBlock
      .findAllMatchIn(ddlText)
      .map { m =>
        val label = m.group(1)
        val props = parseProps(m.group(2))
        label -> NodeType(label, props)
      }
      .toMap

    val rels = RelBlock
      .findAllMatchIn(ddlText)
      .map { m =>
        val edgeLabel = m.group(2)

        val srcRaw = m.group(1) // e.g. "Post | Comment"
        val dstRaw = m.group(4) // e.g. "Person"

        // Simple choice: take the first label (good enough for summarization)
        val srcL = srcRaw.split("\\|").head.trim
        val dstL = dstRaw.split("\\|").head.trim

        val propsBlock = Option(m.group(3)).getOrElse("")
        val props = parseProps(propsBlock)

        edgeLabel -> RelType(edgeLabel, srcL, dstL, props)

      }.toMap

    SchemaCatalog(nodes, rels)
  }

  

  // ----------------------------
  // 3) Constraint schema (Spark)
  // ----------------------------
  // We keep it flexible by reading JSONL and then transforming with Spark SQL.
  // Expected arrays:
  // lhs.nodes: [{v,label,props:[...]}, ...]
  // lhs.edges: [{e,label,src,dst,props:[...]}, ...]
  private val nodePatternSchema = ArrayType(
    StructType(Seq(
      StructField("v", StringType, nullable = true),
      StructField("label", StringType, nullable = true),
      StructField("props", ArrayType(StringType), nullable = true)
    ))
  )

  private val edgePatternSchema = ArrayType(
    StructType(Seq(
      StructField("e", StringType, nullable = true),
      StructField("label", StringType, nullable = true),
      StructField("src", StringType, nullable = true),
      StructField("dst", StringType, nullable = true),
      StructField("props", ArrayType(StringType), nullable = true)
    ))
  )

  private val sideSchema = StructType(Seq(
    StructField("nodes", nodePatternSchema, nullable = true),
    StructField("edges", edgePatternSchema, nullable = true)
  ))

  val constraintSchema = StructType(Seq(
    StructField("cid", StringType, nullable = true),
    StructField("ctype", StringType, nullable = true),
    StructField("lhs", sideSchema, nullable = true),
    StructField("rhs", sideSchema, nullable = true)
  ))

  // ----------------------------
  // 4) Linker logic
  // ----------------------------
  // Output fields added:
  // - linked_ok: boolean
  // - link_errors: array<string>
  // - lhs.nodes_linked: nodes with extra fields (nodeTypeExists, missingProps)
  // - lhs.edges_linked: edges with extra fields (relTypeExists, endpointsCompatible, missingProps)
  // same for rhs.*
  def linkConstraints(constraintsDF: DataFrame, catalog: SchemaCatalog)(implicit spark: SparkSession): (DataFrame, DataFrame) = {
    import spark.implicits._

    val nodePropsMap: Map[String, Set[String]] =
    catalog.nodes.map { case (k, v) => k -> v.props }

    val relInfoMap: Map[String, (String, String, Set[String])] =
    catalog.rels.map { case (k, r) => k -> (r.srcLabel, r.dstLabel, r.props) }

    val bcNodeProps = spark.sparkContext.broadcast(nodePropsMap)
    val bcRelInfo   = spark.sparkContext.broadcast(relInfoMap)

    // Helper UDFs
    val nodeMissingPropsUdf = udf { (label: String, props: Seq[String]) =>
      val p = Option(props).getOrElse(Seq.empty).toSet
      val schemaPropsOpt = bcNodeProps.value.get(label)
      schemaPropsOpt match {
        case None => p.toSeq.sorted // if node type missing, treat all referenced props as "missing"
        case Some(schemaProps) => (p.diff(schemaProps)).toSeq.sorted
      }
    }

    val nodeTypeExistsUdf = udf { (label: String) =>
      bcNodeProps.value.contains(label)
    }

    val relTypeExistsUdf = udf { (label: String) =>
      bcRelInfo.value.contains(label)
    }

    val relMissingPropsUdf = udf { (label: String, props: Seq[String]) =>
      val p = Option(props).getOrElse(Seq.empty).toSet
      bcRelInfo.value.get(label) match {
        case None => p.toSeq.sorted
        case Some((_, _, schemaProps)) => (p.diff(schemaProps)).toSeq.sorted
      }
    }

    // endpointsCompatible: check that edge label's FROM/TO matches node labels of src/dst vars
    // We need a (var -> label) map per constraint side, so we compute it inside a UDF.
    val endpointsCompatibleUdf = udf { (edgeLabel: String, srcVar: String, dstVar: String, sideNodes: Seq[Row]) =>
      val varToLabel: Map[String, String] =
        Option(sideNodes).getOrElse(Seq.empty).flatMap { r =>
          val v = Option(r.getAs[String]("v")).getOrElse("")
          val l = Option(r.getAs[String]("label")).getOrElse("")
          if (v.nonEmpty && l.nonEmpty) Some(v -> l) else None
        }.toMap

      bcRelInfo.value.get(edgeLabel) match {
        case None => false
        case Some((fromL, toL, _)) =>
          val srcL = varToLabel.getOrElse(srcVar, "")
          val dstL = varToLabel.getOrElse(dstVar, "")
          srcL == fromL && dstL == toL
      }
    }


    def linkSide(sideCol: String): (Column, Column, Column) = {
  //val nodesCol = col(s"$sideCol.nodes")
  //val edgesCol = col(s"$sideCol.edges")
  val edgesCol = coalesce(col(s"$sideCol.edges"), array())
  val nodesCol = coalesce(col(s"$sideCol.nodes"), array())

  // linked nodes array
  val nodesLinked =
    transform(nodesCol, n =>
      struct(
        n.getField("v").as("v"),
        n.getField("label").as("label"),
        coalesce(n.getField("props"), array()).as("props"),
        nodeTypeExistsUdf(n.getField("label")).as("nodeTypeExists"),
        nodeMissingPropsUdf(n.getField("label"), n.getField("props")).as("missingProps")
      )
    )

  // linked edges array
  val edgesLinked =
    transform(edgesCol, e =>
      struct(
        e.getField("e").as("e"),
        e.getField("label").as("label"),
        e.getField("src").as("src"),
        e.getField("dst").as("dst"),
        coalesce(e.getField("props"), array()).as("props"),
        relTypeExistsUdf(e.getField("label")).as("relTypeExists"),
        endpointsCompatibleUdf(e.getField("label"), e.getField("src"), e.getField("dst"), nodesCol).as("endpointsCompatible"),
        relMissingPropsUdf(e.getField("label"), e.getField("props")).as("missingProps")
      )
    )

  // IMPORTANT: build errors from nodesCol/edgesCol (not from nodesLinked/edgesLinked)
  val nodeTypeMissing =
    transform(nodesCol, n =>
      when(not(nodeTypeExistsUdf(n.getField("label"))),
        concat(lit(s"$sideCol nodeType missing: "), n.getField("label"), lit(" (var="), n.getField("v"), lit(")"))
      ).otherwise(lit(null))
    )

  val nodePropsMissing =
    transform(nodesCol, n =>
      when(size(nodeMissingPropsUdf(n.getField("label"), n.getField("props"))) > 0,
        concat(
          lit(s"$sideCol node props missing: "),
          n.getField("label"),
          lit(" var="), n.getField("v"),
          lit(" -> "),
          concat_ws(",", nodeMissingPropsUdf(n.getField("label"), n.getField("props")))
        )
      ).otherwise(lit(null))
    )

  val relTypeMissing =
    transform(edgesCol, e =>
      when(not(relTypeExistsUdf(e.getField("label"))),
        concat(lit(s"$sideCol relType missing: "), e.getField("label"), lit(" (edge="), e.getField("e"), lit(")"))
      ).otherwise(lit(null))
    )

  val relEndpointsMismatch =
    transform(edgesCol, e =>
      when(relTypeExistsUdf(e.getField("label")) &&
           not(endpointsCompatibleUdf(e.getField("label"), e.getField("src"), e.getField("dst"), nodesCol)),
        concat(
          lit(s"$sideCol rel endpoints mismatch: "),
          e.getField("label"),
          lit(" srcVar="), e.getField("src"),
          lit(" dstVar="), e.getField("dst"),
          lit(" (edge="), e.getField("e"), lit(")")
        )
      ).otherwise(lit(null))
    )

  val relPropsMissing =
    transform(edgesCol, e =>
      when(size(relMissingPropsUdf(e.getField("label"), e.getField("props"))) > 0,
        concat(
          lit(s"$sideCol rel props missing: "),
          e.getField("label"),
          lit(" edge="), e.getField("e"),
          lit(" -> "),
          concat_ws(",", relMissingPropsUdf(e.getField("label"), e.getField("props")))
        )
      ).otherwise(lit(null))
    )

  val sideErrors =
    array_remove(
      flatten(array(nodeTypeMissing, nodePropsMissing, relTypeMissing, relEndpointsMismatch, relPropsMissing)),
      lit(null)
    )

  (nodesLinked.as(s"${sideCol}_nodes_linked"),
   edgesLinked.as(s"${sideCol}_edges_linked"),
   sideErrors.as(s"${sideCol}_errors"))
}


    /*def linkSide(sideCol: String): (Column, Column, Column) = {
      val nodesCol = col(s"$sideCol.nodes")
      val edgesCol = col(s"$sideCol.edges")

      val nodesLinked =
        transform(nodesCol, n =>
          struct(
            n.getField("v").as("v"),
            n.getField("label").as("label"),
            coalesce(n.getField("props"), array()).as("props"),
            nodeTypeExistsUdf(n.getField("label")).as("nodeTypeExists"),
            nodeMissingPropsUdf(n.getField("label"), n.getField("props")).as("missingProps")
          )
        ).as(s"${sideCol}_nodes_linked")

      val edgesLinked =
        transform(edgesCol, e =>
          struct(
            e.getField("e").as("e"),
            e.getField("label").as("label"),
            e.getField("src").as("src"),
            e.getField("dst").as("dst"),
            coalesce(e.getField("props"), array()).as("props"),
            relTypeExistsUdf(e.getField("label")).as("relTypeExists"),
            endpointsCompatibleUdf(e.getField("label"), e.getField("src"), e.getField("dst"), nodesCol).as("endpointsCompatible"),
            relMissingPropsUdf(e.getField("label"), e.getField("props")).as("missingProps")
          )
        ).as(s"${sideCol}_edges_linked")

      // errors array for this side
      val sideErrors =
        array_remove(flatten(array(
          transform(nodesLinked, ln =>
            when(not(ln.getField("nodeTypeExists")),
              concat(lit(s"$sideCol nodeType missing: "), ln.getField("label"), lit(" (var="), ln.getField("v"), lit(")"))
            ).otherwise(lit(null))
          ),
          transform(nodesLinked, ln =>
            when(size(ln.getField("missingProps")) > 0,
              concat(lit(s"$sideCol node props missing: "), ln.getField("label"), lit(" var="), ln.getField("v"),
                lit(" -> "), concat_ws(",", ln.getField("missingProps")))
            ).otherwise(lit(null))
          ),
          transform(edgesLinked, le =>
            when(not(le.getField("relTypeExists")),
              concat(lit(s"$sideCol relType missing: "), le.getField("label"), lit(" (edge="), le.getField("e"), lit(")"))
            ).otherwise(lit(null))
          ),
          transform(edgesLinked, le =>
            when(le.getField("relTypeExists") && not(le.getField("endpointsCompatible")),
              concat(lit(s"$sideCol rel endpoints mismatch: "), le.getField("label"),
                lit(" srcVar="), le.getField("src"), lit(" dstVar="), le.getField("dst"), lit(" (edge="), le.getField("e"), lit(")"))
            ).otherwise(lit(null))
          ),
          transform(edgesLinked, le =>
            when(size(le.getField("missingProps")) > 0,
              concat(lit(s"$sideCol rel props missing: "), le.getField("label"), lit(" edge="), le.getField("e"),
                lit(" -> "), concat_ws(",", le.getField("missingProps")))
            ).otherwise(lit(null))
          )
        )), lit(null)).as(s"${sideCol}_errors")

      (nodesLinked, edgesLinked, sideErrors)
    }*/

    val (lhsNodesLinked, lhsEdgesLinked, lhsErrors) = linkSide("lhs")
    val (rhsNodesLinked, rhsEdgesLinked, rhsErrors) = linkSide("rhs")

    val linked =
      constraintsDF
        .withColumn("lhs_nodes_linked", lhsNodesLinked)
        .withColumn("lhs_edges_linked", lhsEdgesLinked)
        .withColumn("rhs_nodes_linked", rhsNodesLinked)
        .withColumn("rhs_edges_linked", rhsEdgesLinked)
        .withColumn("link_errors", array_distinct(concat(lhsErrors, rhsErrors)))
        .withColumn("linked_ok", size(col("link_errors")) === 0)

    val errors =
      linked
        .filter(not(col("linked_ok")))
        .select(
          col("cid"),
          col("ctype"),
          explode(col("link_errors")).as("error")
        )

    (linked, errors)
  }

  // ----------------------------
  // 5) Small runnable main
  // ----------------------------
  final case class Config(
    schemaDdlPath: String = "",
    constraintsJsonlPath: String = "",
    outLinkedPath: String = "",
    outErrorsPath: String = "",
    outFormat: String = "parquet" // parquet | json
  )

  def main(args: Array[String]): Unit = {
    // If you already use Typesafe Config + scopt in your project,
    // you can replace this with your existing config loader.
    val cfg = parseArgsOrExit(args)

    implicit val spark: SparkSession =
      SparkSession.builder()
        .appName("PG-SPARC Linker (schema ↔ constraints)")
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val ddlText = spark.read.textFile(cfg.schemaDdlPath).collect().mkString("\n")
    val catalog = parseSchemaDDL(ddlText)

    println(s"Parsed node types: ${catalog.nodes.size}")
    println(s"Parsed edge types: ${catalog.rels.size}")
    println("Edge labels: " + catalog.rels.keys.take(20).mkString(", "))

    // Read constraints JSONL with a known schema (safer than schema inference)
    val constraintsDF =
      spark.read
        .schema(constraintSchema)
        .json(cfg.constraintsJsonlPath)
        .withColumn("lhs", when(col("lhs").isNull, struct(lit(null).as("nodes"), lit(null).as("edges"))).otherwise(col("lhs")))
        .withColumn("rhs", when(col("rhs").isNull, struct(lit(null).as("nodes"), lit(null).as("edges"))).otherwise(col("rhs")))
        .withColumn("lhs", struct(
          coalesce(col("lhs.nodes"), array()).as("nodes"),
          coalesce(col("lhs.edges"), array()).as("edges")
        ))
        .withColumn("rhs", struct(
          coalesce(col("rhs.nodes"), array()).as("nodes"),
          coalesce(col("rhs.edges"), array()).as("edges")
        ))

    val (linkedDF, errorsDF) = linkConstraints(constraintsDF, catalog)

    cfg.outFormat.toLowerCase match {
      case "json" =>
        linkedDF.coalesce(1).write.mode("overwrite").json(cfg.outLinkedPath)
        linkedDF.write.mode("overwrite").csv(cfg.outLinkedPath)
        errorsDF.coalesce(1).write.mode("overwrite").json(cfg.outErrorsPath)
      case _ =>
        linkedDF.write.mode("overwrite").parquet(cfg.outLinkedPath)
        errorsDF.write.mode("overwrite").parquet(cfg.outErrorsPath)
    }

    linkedDF
      .coalesce(1)
      .write
      .mode("overwrite")
      .json(cfg.outLinkedPath)


    // Helpful console summary
    val total = linkedDF.count()
    val ok    = linkedDF.filter(col("linked_ok")).count()
    val bad   = total - ok
    println(s"[Linker] total=$total, linked_ok=$ok, linked_failed=$bad")
    println(s"[Linker] wrote linked -> ${cfg.outLinkedPath}")
    println(s"[Linker] wrote errors -> ${cfg.outErrorsPath}")

    spark.stop()
  }

  // ----------------------------
  // 6) Minimal arg parser (no dependencies)
  // ----------------------------
  private def parseArgsOrExit(args: Array[String]): Config = {
    def get(flag: String): Option[String] = {
      val idx = args.indexOf(flag)
      if (idx >= 0 && idx + 1 < args.length) Some(args(idx + 1)) else None
    }

    val cfg = Config(
      schemaDdlPath = get("--schema").getOrElse(""),
      constraintsJsonlPath = get("--constraints").getOrElse(""),
      outLinkedPath = get("--outLinked").getOrElse(""),
      outErrorsPath = get("--outErrors").getOrElse(""),
      outFormat = get("--format").getOrElse("parquet")
    )

    val missing =
      Seq("--schema" -> cfg.schemaDdlPath,
          "--constraints" -> cfg.constraintsJsonlPath,
          "--outLinked" -> cfg.outLinkedPath,
          "--outErrors" -> cfg.outErrorsPath).collect { case (k, v) if v.trim.isEmpty => k }

    if (missing.nonEmpty) {
      System.err.println(
        s"""
           |Missing arguments: ${missing.mkString(", ")}
           |
           |Usage:
           |  spark-submit ... pgpartitioner.Linker
           |    --schema /path/schema.ddl
           |    --constraints /path/constraints.jsonl
           |    --outLinked /path/out/linked_constraints
           |    --outErrors /path/out/link_errors
           |    [--format parquet|json]
           |
           |Outputs:
           |  linked_constraints: original constraint + linked_ok + link_errors + *_nodes_linked + *_edges_linked
           |  link_errors: (cid, ctype, error)
           |""".stripMargin
      )
      System.exit(1)
    }
    cfg
  }
}