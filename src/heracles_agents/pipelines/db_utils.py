import re
from heracles.query_interface import Neo4jWrapper

def clean_cypher_string(cypher_string: str) -> str:
    """
    Removes Markdown code block encapsulation from a cypher query if present.
    """
    # Regex matches ```cypher ... ``` or ``` ... ```
    pattern = r"```(?:cypher)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, cypher_string, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cypher_string.strip()

def query_db(dsgdb_conf, cypher_string):
    cypher_string = clean_cypher_string(cypher_string)

    with Neo4jWrapper(
        dsgdb_conf.uri,
        (
            dsgdb_conf.username.get_secret_value(),
            dsgdb_conf.password.get_secret_value(),
        ),
        atomic_queries=True,
        print_profiles=False,
    ) as db:
        if dsgdb_conf.n_object_verification is not None:
            v = db.query("MATCH (n:Object) RETURN COUNT(*) as count")
            count = v[0]["count"]
            assert count == dsgdb_conf.n_object_verification, (
                f"Connected database has {count} objects "
                f"({dsgdb_conf.n_object_verification} expected)"
            )
        try:
            query_result = str(db.query(cypher_string))
            return True, query_result
        except Exception as ex:
            print(ex)
            query_result = str(ex)
            return False, query_result