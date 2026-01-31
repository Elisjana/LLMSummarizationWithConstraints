package dataparserModule

import java.nio.file.{Files, Paths, Path}
import java.nio.charset.StandardCharsets
import scala.util.Try
import scala.collection.mutable
import scala.collection.JavaConverters._
import ujson._
import utils.CommonUtils


object ConstraintSerializer {

//################################################################### Create Json schema structure ###################################################################
    case class Node(varName: String, label: String)
    case class Edge(varName: String, label: String, src: String, dst: String)
    case class Side(nodes: Vector[Node], edges: Vector[Edge])


    case class ConstraintRec(
        id: String,
        dataset: String,
        ctype: String,    
        lhs: Side,
        rhs: Option[Side] 
    )

    //################################################################### Add regex patterns for parsing constraint TXT files ###################################################################
    private val nodeRe   = """(?i)^\s*Node:\s*([A-Za-z0-9_]+)\s+Label:([A-Za-z0-9_]+)\s*$""".r
    private val edgeRe   = """(?i)^\s*Edge:\s*([A-Za-z0-9_]+)\s+Label:([A-Za-z0-9_]+)\s+Source:([A-Za-z0-9_]+)\s+Target:([A-Za-z0-9_]+)\s*$""".r
    private val lhsHdr   = """(?i)^----\s*Source\s+Graph\s+Pattern\s*:\s*----\s*$""".r
    private val rhsHdr   = """(?i)^--------\s*Target\s+Graph\s+Pattern\s*--------\s*$""".r
    private val ignoreHr = """(?i)^\s*-{2,}\s*Source\s+Constraints\s*-{2,}\s*$""".r

    //#################################################### Converts identifiers into uniform variable names #######################################################
    private def toVar(raw: String): String = "n" + raw.replaceAll("""[^A-Za-z0-9_]""", "_")

    //#################################################### Parse TXT lines and convert nodes and edges to objects ################################################
    private def parseSide(lines: Vector[String]): Side = {
        val nodes = mutable.LinkedHashMap[String, Node]()
        val edges = mutable.ArrayBuffer[Edge]()

        lines.foreach {
            case nodeRe(id, label) =>
                val v = toVar(id); nodes.getOrElseUpdate(v, Node(v, label))

            case edgeRe(evar, label, src, dst) =>
                edges += Edge(evar, CommonUtils.normalizeRelLabel(label), toVar(src), toVar(dst))
            case _ => 
        }
        Side(nodes.values.toVector, edges.toVector)
    }

    //################################################# Split sections, remove irrelevant data, and extract [LHS, RHS] ############################################

    private def splitSections(all: Vector[String]): (Vector[String], Vector[String]) = {
        val lhsStart = all.indexWhere { case lhsHdr() => true; case _ => false }
        val rhsStart = all.indexWhere { case rhsHdr() => true; case _ => false }
        if (lhsStart == -1) sys.error("Missing '----Source Graph Pattern----' header in TXT file.")
        val lhsBlock = {
        val end = if (rhsStart == -1) all.length else rhsStart
        all.slice(lhsStart + 1, end)
            .filterNot(s => s.trim.isEmpty || ignoreHr.findFirstIn(s).nonEmpty)
        }
        val rhsBlock =
        if (rhsStart == -1) Vector.empty[String]
        else all.drop(rhsStart + 1).filterNot(_.trim.isEmpty)
        (lhsBlock, rhsBlock)
    }

    private def listChildren(dir: Path): Vector[Path] =
        if (!Files.isDirectory(dir)) Vector.empty
        else Files.list(dir).iterator().asScala.toVector

    private def listTxt(dir: Path): Vector[Path] =
        listChildren(dir).filter(p => Files.isRegularFile(p) && p.getFileName.toString.toLowerCase.endsWith(".txt"))

    //######################################################### Convert txt constraints to JSON #####################################################

    private def constraintToJson(c: ConstraintRec): Value = {
        val lhsNodes = c.lhs.nodes.map(n => Obj("var" -> n.varName, "Node label" -> n.label))
        val lhsEdges = c.lhs.edges.map(e => Obj("var" -> e.varName, "Edge label" -> e.label, "src" -> e.src, "dst" -> e.dst))

        val obj = Obj(
        "id"            -> c.id,
        "dataset"       -> c.dataset,
        "type"          -> c.ctype,
        "lhs"           -> Obj("nodes" -> Arr(lhsNodes:_*), "edges" -> Arr(lhsEdges:_*))
        )

        c.rhs.foreach { rhs =>
            val rhsNodes = rhs.nodes.map(n => Obj("var" -> n.varName, "Node label" -> n.label))
            val rhsEdges = rhs.edges.map(e => Obj("var" -> e.varName, "Edge label" -> e.label, "src" -> e.src, "dst" -> e.dst))
            obj("rhs") = Obj("nodes" -> Arr(rhsNodes:_*), "edges" -> Arr(rhsEdges:_*))
        }
        obj
    }
    
    //######################################################### Process dataset folders i.e., LDBC #####################################################

    def processDataset(datasetDir: Path, version: String, outRoot: Path): Unit = {
        val dataset = datasetDir.getFileName.toString
        val ggdDir  = datasetDir.resolve("GGDs")
        val gfdDir  = datasetDir.resolve("GFDs")
        val pkDir   = datasetDir.resolve("PG-KEYs")

        val targets: Vector[(String, Path)] =
        listTxt(ggdDir).map("GGD"  -> _) ++
        listTxt(gfdDir).map("GFD"  -> _) ++
        listTxt(pkDir ).map("PGKEY"-> _)

        if (targets.isEmpty) {
        println(s"[WARN] No TXT files under $datasetDir (skipping).")
        return
        }

        val outDir   = outRoot.resolve(dataset.toLowerCase.replaceAll("""\s+""","-")).resolve(version.toLowerCase)
        Files.createDirectories(outDir)
        val outJsonl = outDir.resolve("constraints.jsonl")
        val outMan   = outDir.resolve("manifest.json")

        val linesOut = mutable.ArrayBuffer[String]()
        val counts   = mutable.Map("GGD"->0, "GFD"->0, "PGKEY"->0).withDefaultValue(0)
        var idx = 1

        targets.sortBy(_._2.toString).foreach { case (ctype, path) =>
        val raw = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
        val (lhsLines, rhsLines) = splitSections(raw)
        val lhs = parseSide(lhsLines)
        val rhs = if (ctype == "GGD" && rhsLines.nonEmpty) Some(parseSide(rhsLines)) else None
        val id = s"${ctype}_${"%04d".format(idx)}"

        val rec = ConstraintRec(
            id = id,
            dataset = dataset,
            ctype = ctype,
            lhs = lhs,
            rhs = rhs
        )

        val json = constraintToJson(rec)
        linesOut += ujson.write(json) 
        counts.update(ctype, counts(ctype) + 1)
        idx += 1
    }

    Files.write(outJsonl, linesOut.mkString("\n").getBytes(StandardCharsets.UTF_8))
    println(s"[OK] ${dataset}: wrote ${linesOut.size} constraints → $outJsonl")

    val manifest = Obj(
        "dataset"       -> dataset,
        "graph_version" -> version,
        "created_at"    ->  java.time.ZonedDateTime.now().toString,
        "source_root"   -> datasetDir.toString,
        "counts"        -> Obj(
            "GGD"   -> counts("GGD"),
            "GFD"   -> counts("GFD"),
            "PGKEY" -> counts("PGKEY")
        )
    )

    Files.write(outMan, ujson.write(manifest, indent = 2).getBytes(StandardCharsets.UTF_8))
        println(s"[OK] ${dataset}: wrote manifest → $outMan")
    }

    //################################################################### Process *all* datasets under an input root ###################################################################

    def processAll(inputRoot: Path, version: String, outRoot: Path): Unit = {
        if (!Files.isDirectory(inputRoot)) {
            System.err.println(s"[ERROR] Input root not found: $inputRoot")
            return
        }

        val datasets =
        listChildren(inputRoot)
            .filter(Files.isDirectory(_))
            .sortBy(_.getFileName.toString)

        if (datasets.isEmpty) {
            System.err.println(s"[WARN] No dataset folders under " + inputRoot)
            return
        }

        datasets.foreach { ds =>
            println(s"[INFO] → ${ds.getFileName}")
            processDataset(ds, version, outRoot)   
        }
        
    }

}

