package pgsummaryllm

import org.apache.spark.sql.SparkSession
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.io.Source
import scala.util.Try
import ujson._

/**
 * Build Gemma JSONL inputs for "schema + constraints aware summarization"
 * with: ONE JSONL RECORD PER CONSTRAINT (GGD/GFD/PG-KEY line).
 *
 * Each output line is:
 * {
 *   "id": "<constraint-id>",
 *   "instruction": "...",
 *   "context": "SCHEMA + this constraint (+ optional small schema digest)",
 *   "input": "Summarize this constraint in schema-aware terms...",
 *   "output": ""
 * }
 */
object GemmaInputBuilder {

  // -----------------------------
  // Config
  // -----------------------------
  final case class Config(
    schemaPath: String,
    constraintsJsonlPath: String,

    // Caps
    maxConstraintsTotal: Int = 5000,        // hard cap total constraints processed
    maxConstraintsPerType: Int = 5000,      // cap per type (GGD/GFD/PG-KEY)
    maxConstraintsPerUnknown: Int = 5000,   // cap unknown

    // Schema verbosity controls
    includePrettySchemaJson: Boolean = false, // usually false (too long)
    maxSchemaPrettyChars: Int = 6000,         // if includePrettySchemaJson=true

    // Constraint formatting / prompt size controls
    maxSideNodes: Int = 12,
    maxSideEdges: Int = 18,
    maxLinkErrors: Int = 8,

    // Optional label normalization (align constraint labels with schema)
    normalizeEdgeLabels: Boolean = true,

    // Output
    outJsonlPath: String
  )

  // -----------------------------
  // Label normalization
  // -----------------------------
  // Your schema uses WORKS_AT / STUDIES_AT, but constraints sometimes use WORK_AT / STUDY_AT.
  private val edgeLabelNormalize: Map[String, String] = Map(
    "WORK_AT"  -> "WORKS_AT",
    "STUDY_AT" -> "STUDIES_AT"
  )

  private def normEdgeLabel(s: String, enabled: Boolean): String =
    if (!enabled) s else edgeLabelNormalize.getOrElse(s, s)

  // -----------------------------
  // Public API
  // -----------------------------
  def build(spark: SparkSession, cfg: Config): Unit = {
    // spark is not required for this variant, but kept in signature to match your pipeline style.
    val schemaBundle = loadSchemaBundle(cfg.schemaPath, cfg)

    val constraints = loadConstraints(cfg.constraintsJsonlPath, cfg)

    val records: Seq[ujson.Value] = constraints.map { c =>
      val cid   = constraintId(c)
      val ctype = constraintType(c)

      val constraintText = constraintToText(c, cfg)

      val context =
        s"""|You are given a property graph dataset description.
            |
            |=== SCHEMA (authoritative) ===
            |${schemaBundle}
            |
            |=== SINGLE CONSTRAINT (authoritative) ===
            |$constraintText
            |""".stripMargin

      val instruction =
        """You are a graph summarization assistant.
          |Produce a compact, faithful natural-language explanation of ONE constraint, grounded in the schema.
          |You MUST respect schema terminology (node labels, edge labels, property names).
          |You MUST NOT invent labels/properties not present in the schema.
          |You MUST preserve the meaning of the constraint.
          |Output MUST be understandable to a data engineer reading the schema.
          |""".stripMargin

      val input =
        s"""Summarize constraint $cid ($ctype).
           |Include:
           |1) Plain-language meaning (what pattern it checks/creates),
           |2) Which entity types and relationships it involves (use schema labels),
           |3) The "expected/required pattern" implied by the rule (bullet list),
           |4) Any data quality/consistency implication if violated.
           |""".stripMargin

      Obj(
        "id" -> Str(cid),
        "instruction" -> Str(instruction.trim),
        "context" -> Str(context.trim),
        "input" -> Str(input.trim),
        "output" -> Str("")
      )
    }

    writeJsonl(cfg.outJsonlPath, records)
    println(s"[GemmaInputBuilder] Wrote ${records.size} JSONL records to: ${cfg.outJsonlPath}")
  }

  // -----------------------------
  // Schema handling
  // -----------------------------
  private def loadSchemaBundle(schemaPath: String, cfg: Config): String = {
    val raw = Source.fromFile(schemaPath, "UTF-8").mkString
    val trimmed = raw.trim

    if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) {
      // plain text schema
      return raw
    }

    Try {
      val js = ujson.read(trimmed)
      val digest = schemaDigest(js, cfg)
      if (!cfg.includePrettySchemaJson) digest
      else {
        val pretty = ujson.write(js, indent = 2)
        val clipped =
          if (pretty.length <= cfg.maxSchemaPrettyChars) pretty
          else pretty.take(cfg.maxSchemaPrettyChars) + "\n... (clipped) ..."
        s"""|$digest
            |
            |--- schema.json (pretty; clipped) ---
            |$clipped
            |""".stripMargin
      }
    }.getOrElse(raw)
  }

  /**
   * UPDATED for your schema shape:
   * {
   *   "nodes": { "PersonType": { "label":"Person", "properties":[...] }, ... },
   *   "edges": { "KNOWS": { "from":[...], "to":[...], "properties":[...] }, ... }
   * }
   */
  private def schemaDigest(js: ujson.Value, cfg: Config): String = {
    val nodesObj = js.obj.get("nodes").collect { case o: ujson.Obj => o }.getOrElse(ujson.Obj())
    val edgesObj = js.obj.get("edges").collect { case o: ujson.Obj => o }.getOrElse(ujson.Obj())

    val nodeLines =
      if (nodesObj.value.nonEmpty) {
        val lines = nodesObj.value.toSeq.take(120).map { case (typeName, defn) =>
          val label = defn.obj.get("label").map(_.str).getOrElse(typeName)
          val props = defn.obj.get("properties") match {
            case Some(a: ujson.Arr) =>
              a.value.take(14).map { p =>
                val n = p.obj.get("name").map(_.str).getOrElse("?")
                val t = p.obj.get("type").map(_.str).getOrElse("?")
                val opt = p.obj.get("optional").map(_.bool).getOrElse(false)
                if (opt) s"$n:$t?" else s"$n:$t"
              }.mkString(", ")
            case _ => ""
          }
          if (props.nonEmpty) s"- $label { $props, ... }" else s"- $label"
        }
        "Node types:\n" + lines.mkString("\n")
      } else "Node types: (not extracted)\n"

    val edgeLines =
      if (edgesObj.value.nonEmpty) {
        val lines = edgesObj.value.toSeq.take(200).map { case (edgeLabelRaw, defn) =>
          val edgeLabel = normEdgeLabel(edgeLabelRaw, cfg.normalizeEdgeLabels)

          val from = defn.obj.get("from").collect {
            case a: ujson.Arr => a.value.map(_.str).mkString("|")
          }.getOrElse("?")

          val to = defn.obj.get("to").collect {
            case a: ujson.Arr => a.value.map(_.str).mkString("|")
          }.getOrElse("?")

          val props = defn.obj.get("properties") match {
            case Some(a: ujson.Arr) if a.value.nonEmpty =>
              a.value.take(10).map { p =>
                val n = p.obj.get("name").map(_.str).getOrElse("?")
                val t = p.obj.get("type").map(_.str).getOrElse("?")
                val opt = p.obj.get("optional").map(_.bool).getOrElse(false)
                if (opt) s"$n:$t?" else s"$n:$t"
              }.mkString(", ")
            case _ => ""
          }

          if (props.nonEmpty) s"- $edgeLabel ($from -> $to) { $props }"
          else s"- $edgeLabel ($from -> $to)"
        }
        "Edge types:\n" + lines.mkString("\n")
      } else "Edge types: (not extracted)\n"

    s"$nodeLines\n\n$edgeLines"
  }

  // -----------------------------
  // Constraints handling (JSONL)
  // -----------------------------
  private def loadConstraints(path: String, cfg: Config): Vector[ujson.Value] = {
    val lines = Source.fromFile(path, "UTF-8").getLines().toVector.filter(_.trim.nonEmpty)

    val parsed: Vector[ujson.Value] =
      lines.flatMap(ln => Try(ujson.read(ln)).toOption)

    val grouped: Map[String, Vector[ujson.Value]] =
      parsed.groupBy(constraintType)

    // Apply caps per type, then global cap
    val selectedPerType: Vector[ujson.Value] =
      grouped.toSeq.sortBy(_._1).flatMap { case (t, vec) =>
        val cap =
          if (t == "UNKNOWN") cfg.maxConstraintsPerUnknown
          else cfg.maxConstraintsPerType
        vec.take(cap)
      }.toVector

    selectedPerType.take(cfg.maxConstraintsTotal)
  }

  private def constraintId(c: ujson.Value): String =
    c.obj.get("id").map(_.str)
      .orElse(c.obj.get("cid").map(_.str))
      .getOrElse("constraint_unknown")

  private def constraintType(c: ujson.Value): String =
    c.obj.get("type").map(_.str)
      .orElse(c.obj.get("ctype").map(_.str))
      .getOrElse("UNKNOWN")

  /**
   * Converts ONE constraint JSON into a compact, schema-aware text.
   * (Uses summary if exists; else constructs from lhs/rhs.)
   */
  private def constraintToText(c: ujson.Value, cfg: Config): String = {
    val cid = constraintId(c)
    val ctype = constraintType(c)

    val summary =
      c.obj.get("summary").map(_.str).getOrElse {
        val lhs = formatSide(c.obj.get("lhs"), cfg)
        val rhs = formatSide(c.obj.get("rhs"), cfg)
        if (lhs.nonEmpty || rhs.nonEmpty) s"If $lhs then $rhs" else "(no summary)"
      }

    val linkErrors =
      c.obj.get("link_errors") match {
        case Some(arr: ujson.Arr) if arr.value.nonEmpty =>
          arr.value.take(cfg.maxLinkErrors).map(_.str).mkString("; ")
        case _ => ""
      }

    val le = if (linkErrors.nonEmpty) s"\nlink_errors: $linkErrors" else ""

    s"""|[$ctype] $cid
        |$summary$le
        |""".stripMargin.trim
  }

  private def formatSide(sideOpt: Option[ujson.Value], cfg: Config): String = sideOpt match {
    case None => ""
    case Some(side) =>
      val nodes = side.obj.get("nodes") match {
        case Some(a: ujson.Arr) =>
          a.value.take(cfg.maxSideNodes).map { n =>
            val v = n.obj.get("var").map(_.str).getOrElse("?")
            val lbl =
              n.obj.get("Node label").map(_.str)
                .orElse(n.obj.get("label").map(_.str))
                .getOrElse("?")
            s"$v:$lbl"
          }.mkString(", ")
        case _ => ""
      }

      val edges = side.obj.get("edges") match {
        case Some(a: ujson.Arr) =>
          a.value.take(cfg.maxSideEdges).map { e =>
            val raw =
              e.obj.get("Edge label").map(_.str)
                .orElse(e.obj.get("label").map(_.str))
                .getOrElse("?")

            val elbl = normEdgeLabel(raw, cfg.normalizeEdgeLabels)

            val s = e.obj.get("src").map(_.str).getOrElse("?")
            val d = e.obj.get("dst").map(_.str).getOrElse("?")
            s"$elbl($s->$d)"
          }.mkString(", ")
        case _ => ""
      }

      val parts = Seq(
        if (nodes.nonEmpty) s"Nodes: $nodes" else "",
        if (edges.nonEmpty) s"Edges: $edges" else ""
      ).filter(_.nonEmpty)

      parts.mkString(" | ")
  }

  // -----------------------------
  // JSONL writer
  // -----------------------------
  private def writeJsonl(outPath: String, records: Seq[ujson.Value]): Unit = {
    val out = records.map(r => ujson.write(r)).mkString("\n") + "\n"
    val p = Paths.get(outPath)
    if (p.getParent != null) Files.createDirectories(p.getParent)
    Files.write(p, out.getBytes(StandardCharsets.UTF_8))
  }
}