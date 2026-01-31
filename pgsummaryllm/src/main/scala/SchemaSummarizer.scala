package pgsummaryllm

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object SchemaSummarizer {

  /** Summarize node types using schema only + linked constraints */
  def summarizeNodeTypes(
      catalog: Linker.SchemaCatalog,
      linkedConstraintsDF: DataFrame
  )(implicit spark: SparkSession): DataFrame = {

    import spark.implicits._

    // Build: label -> constraint tags like "gfd(c1)", "ggd(c2)"
    val nodeConstraints =
      linkedConstraintsDF
        .filter(col("linked_ok") === true)
        .select(col("cid"), col("ctype"), explode_outer(col("lhs_nodes_linked")).as("n"))
        .select(col("cid"), col("ctype"), col("n.label").as("label"))
        .unionByName(
          linkedConstraintsDF
            .filter(col("linked_ok") === true)
            .select(col("cid"), col("ctype"), explode_outer(col("rhs_nodes_linked")).as("n"))
            .select(col("cid"), col("ctype"), col("n.label").as("label"))
        )
        .filter(col("label").isNotNull)
        .groupBy("label")
        .agg(collect_set(concat(col("ctype"), lit("("), col("cid"), lit(")"))).as("constraint_tags"))

    val nodeTypesDF =
      catalog.nodes.values.toSeq
        .map(nt => (nt.label, nt.props.toSeq.sorted))
        .toDF("label", "properties")

    nodeTypesDF
      .join(nodeConstraints, Seq("label"), "left")
      .withColumn("constraint_tags", coalesce(col("constraint_tags"), array()))
      .withColumn(
        "schema_summary",
        concat(
          lit("NodeType "),
          col("label"),
          lit(" with properties: "),
          when(size(col("properties")) > 0, concat_ws(", ", col("properties")))
            .otherwise(lit("(none)")),
          when(size(col("constraint_tags")) > 0,
            concat(lit(". Used in constraints: "), concat_ws(", ", col("constraint_tags")))
          ).otherwise(lit(""))
        )
      )
  }

  /** Summarize relationship types using schema only + linked constraints */
  def summarizeRelTypes(
      catalog: Linker.SchemaCatalog,
      linkedConstraintsDF: DataFrame
  )(implicit spark: SparkSession): DataFrame = {

    import spark.implicits._

    val relConstraints =
      linkedConstraintsDF
        .filter(col("linked_ok") === true)
        .select(col("cid"), col("ctype"), explode_outer(col("lhs_edges_linked")).as("e"))
        .select(col("cid"), col("ctype"), col("e.label").as("relLabel"))
        .unionByName(
          linkedConstraintsDF
            .filter(col("linked_ok") === true)
            .select(col("cid"), col("ctype"), explode_outer(col("rhs_edges_linked")).as("e"))
            .select(col("cid"), col("ctype"), col("e.label").as("relLabel"))
        )
        .filter(col("relLabel").isNotNull)
        .groupBy("relLabel")
        .agg(collect_set(concat(col("ctype"), lit("("), col("cid"), lit(")"))).as("constraint_tags"))

    val relTypesDF =
      catalog.rels.values.toSeq
        .map(rt => (rt.label, rt.srcLabel, rt.dstLabel, rt.props.toSeq.sorted))
        .toDF("relLabel", "fromLabel", "toLabel", "properties")

    relTypesDF
      .join(relConstraints, Seq("relLabel"), "left")
      .withColumn("constraint_tags", coalesce(col("constraint_tags"), array()))
      .withColumn(
        "schema_summary",
        concat(
          lit("RelType "),
          col("relLabel"),
          lit(": "),
          col("fromLabel"),
          lit(" -> "),
          col("toLabel"),
          lit(". Properties: "),
          when(size(col("properties")) > 0, concat_ws(", ", col("properties")))
            .otherwise(lit("(none)")),
          when(size(col("constraint_tags")) > 0,
            concat(lit(". Used in constraints: "), concat_ws(", ", col("constraint_tags")))
          ).otherwise(lit(""))
        )
      )
  }
}