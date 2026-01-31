package pgsummaryllm

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object ConstraintSummarizer {

  /**
   * Schema-only summarization per constraint.
   * Input: linked constraints DF from Linker (parquet written by Step 1)
   * Output: (cid, ctype, linked_ok, summary, link_errors)
   */
  def summarizeConstraints(linkedDF: DataFrame)(implicit spark: SparkSession): DataFrame = {

    // 0) Drop malformed JSONL records that produce null cid/ctype
    val base =
      linkedDF
        .filter(col("cid").isNotNull && col("ctype").isNotNull)

    // Helpers to render one node-pattern / edge-pattern as text
    // Node struct fields expected from Linker output:
    //   v, label, props, nodeTypeExists, missingProps
    val renderNodes = (colName: String) => {
      val n = col(colName)
      // "Person{id,firstName}" (schema-only)
      val txt =
        concat_ws(
          ", ",
          transform(
            coalesce(n, array()),
            x =>
              concat(
                coalesce(x.getField("label"), lit("Unknown")),
                when(size(coalesce(x.getField("props"), array())) > 0,
                  concat(lit("{"), concat_ws(",", x.getField("props")), lit("}"))
                ).otherwise(lit(""))
              )
          )
        )
      // if empty string, return ""
      txt
    }

    // Edge struct fields expected:
    //   e, label, src, dst, props, relTypeExists, endpointsCompatible, missingProps
    val renderEdges = (colName: String) => {
      val e = col(colName)
      // "KNOWS(srcVar->dstVar){prop1,prop2}"
      val txt =
        concat_ws(
          ", ",
          transform(
            coalesce(e, array()),
            x =>
              concat(
                coalesce(x.getField("label"), lit("REL")),
                lit("("),
                coalesce(x.getField("src"), lit("?")),
                lit("->"),
                coalesce(x.getField("dst"), lit("?")),
                lit(")"),
                when(size(coalesce(x.getField("props"), array())) > 0,
                  concat(lit("{"), concat_ws(",", x.getField("props")), lit("}"))
                ).otherwise(lit(""))
              )
          )
        )
      txt
    }

    val lhsNodesTxt = renderNodes("lhs_nodes_linked")
    val rhsNodesTxt = renderNodes("rhs_nodes_linked")
    val lhsEdgesTxt = renderEdges("lhs_edges_linked")
    val rhsEdgesTxt = renderEdges("rhs_edges_linked")

    // Build readable “pattern” text for LHS/RHS, and handle empty
    val lhsPatternRaw =
      trim(
        concat_ws(
          " ",
          when(length(lhsNodesTxt) > 0, concat(lit("Nodes: "), lhsNodesTxt)).otherwise(lit("")),
          when(length(lhsEdgesTxt) > 0, concat(lit("Edges: "), lhsEdgesTxt)).otherwise(lit(""))
        )
      )

    val rhsPatternRaw =
      trim(
        concat_ws(
          " ",
          when(length(rhsNodesTxt) > 0, concat(lit("Nodes: "), rhsNodesTxt)).otherwise(lit("")),
          when(length(rhsEdgesTxt) > 0, concat(lit("Edges: "), rhsEdgesTxt)).otherwise(lit(""))
        )
      )

    val lhsPattern = when(length(lhsPatternRaw) === 0, lit("(empty)")).otherwise(lhsPatternRaw)
    val rhsPattern = when(length(rhsPatternRaw) === 0, lit("(empty)")).otherwise(rhsPatternRaw)

    // Type-specific natural language prefix
    val prefix =
      when(lower(col("ctype")) === "gfd", lit("gFD"))
        .when(lower(col("ctype")) === "ggd", lit("gGD"))
        .when(lower(col("ctype")) === "pgkey", lit("PG-KEY"))
        .otherwise(concat(lit("Constraint("), col("ctype"), lit(")")))

    // Implication phrasing
    val implication =
      when(lower(col("ctype")).isin("gfd", "pgkey"),
        concat(
          prefix, lit(" "), col("cid"),
          lit(": "), lhsPattern,
          lit("  ⇒  "), rhsPattern
        )
      ).otherwise(
        concat(
          prefix, lit(" "), col("cid"),
          lit(": If "), lhsPattern,
          lit(", then "), rhsPattern
        )
      )

    // If not linked_ok, mention errors
    val summaryCol =
      when(col("linked_ok") === true, implication)
        .otherwise(
          concat(
            implication,
            lit(". Link errors: "),
            concat_ws(" | ", coalesce(col("link_errors"), array()))
          )
        )

    base
      .select(
        col("cid"),
        col("ctype"),
        col("linked_ok"),
        col("link_errors"),
        col("lhs_nodes_linked"),
        col("lhs_edges_linked"),
        col("rhs_nodes_linked"),
        col("rhs_edges_linked")
      )
      .withColumn("summary", summaryCol)
      .select("cid", "ctype", "linked_ok", "summary", "link_errors")
  }
}