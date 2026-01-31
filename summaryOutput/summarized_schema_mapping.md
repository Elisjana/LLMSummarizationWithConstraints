# Constraint-Preserving Schema Summary (B1)

- Model: `gemma2:2b`
- Input: `JsonOutput/ldbc/v1/constraints.jsonl`
- GGDs preserved: **221/221**

## Node types
- Comment
- Forum
- Organisation
- Person
- Place
- Post
- Tag

## Edge types (LLM meanings, fallback only if necessary)
- CONTAINER_OF (Forum → Post) — Forum CONTAINER_OF Post
- HAS_CREATOR (Comment → Person) — Comment HAS_CREATOR Person
- HAS_CREATOR (Post → Person) — Post HAS_CREATOR Person
- HAS_INTEREST (Person → Tag) — Person HAS_INTEREST Tag
- HAS_MEMBER (Forum → Person) — Forum HAS_MEMBER Person
- HAS_MODERATOR (Forum → Person) — Forum HAS_MODERATOR Person
- HAS_TAG (Comment → Tag) — Comment HAS_TAG Tag
- HAS_TAG (Forum → Tag) — Forum HAS_TAG Tag
- HAS_TAG (Post → Tag) — Post HAS_TAG Tag
- IS_LOCATED_IN (Comment → Place) — Comment IS_LOCATED_IN Place
- IS_LOCATED_IN (Organisation → Place) — Organisation IS_LOCATED_IN Place
- IS_LOCATED_IN (Person → Place) — Person IS_LOCATED_IN Place
- IS_LOCATED_IN (Post → Place) — Post IS_LOCATED_IN Place
- IS_PART_OF (Place → Place) — Place IS_PART_OF Place
- KNOWS (Person → Person) — Person KNOWS Person
- LIKES (Person → Comment) — Person LIKES Comment
- LIKES (Person → Post) — Person LIKES Post
- REPLY_OF (Comment → Comment) — Comment REPLY_OF Comment
- REPLY_OF (Comment → Post) — A comment replies to a post.
- STUDY_AT (Person → Organisation) — A person studies at an organisation.
- WORK_AT (Person → Organisation) — A person works at an organisation.

## Schema summary text
This schema describes a property graph representing social interactions and information sharing. Nodes represent entities like 'Comment', 'Forum', 'Person', or 'Place'. Edges connect these nodes, indicating relationships such as 'CONTAINER_OF' (a forum contains posts), 'HAS_CREATOR' (comments are created by people), 'HAS_INTEREST' (people have interests in tags), and 'IS_LOCATED_IN' (places can be located within other places). The schema also captures connections like 'KNOWS', 'LIKES', and 'REPLY_OF' to represent social interactions, knowledge sharing, and responses.

## Constraint preservation (canonical signatures)
- ggds_total: 221
- ggds_preserved: 221

(See JSON for full preserved_rules list.)
