package utils

import com.typesafe.config.{Config, ConfigFactory}


/*object QueryTuning {
    private val root: Config = ConfigFactory.load()
    private val cfg:  Config = root.getConfig("query-tuning.neo4j")

    val MaxBatchSize: Int  = cfg.getInt("maxBatchSize")
    val Limit: Int    = cfg.getInt("limit")
    val TxTimeoutMs: Long  = cfg.getLong("tx-timeout-ms")
    val MaxRetries: Int    = cfg.getInt("max-retries")

}*/

object CommonUtils {

    def normalizeRelLabel(label: String): String = {

        val s = label.trim
        val low = s.toLowerCase

        if (low.startsWith("hasmoderator"))      "HAS_MODERATOR"
        else if (low.startsWith("hasmember"))    "HAS_MEMBER"
        else if (low.startsWith("hascreator"))   "HAS_CREATOR"
        else if (low.startsWith("replyof"))      "REPLY_OF"
        else if (low.startsWith("knows"))        "KNOWS"
        else if (low.startsWith("hasinterest"))  "HAS_INTEREST"
        else if (low.startsWith("likes"))        "LIKES"
        else if (low.startsWith("studyat"))      "STUDY_AT"
        else if (low.startsWith("containerof"))  "CONTAINER_OF"
        else if (low.startsWith("workat"))       "WORK_AT"
        else if (low.startsWith("hastag"))       "HAS_TAG"
        else if (low.startsWith("islocatedin"))  "IS_LOCATED_IN"
        else if (low.startsWith("ispartof"))     "IS_PART_OF"
        else if (low.startsWith("hastype"))      "HAS_TYPE"
        else if (low.startsWith("issubclassof")) "IS_SUBCLASS_OF"

        else s
    }

}