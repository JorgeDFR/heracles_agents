from heracles.query_interface import Neo4jWrapper

from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_agents.tool_calling.tool_description import FunctionParameter, ToolDescription
from heracles_agents.tool_calling.registry import ToolRegistry, register_tool

import logging
logger = logging.getLogger(__name__)


class CypherQueryResult(str):
    """String-compatible tool result with execution evidence for evaluation."""

    def __new__(cls, value, *, rows=None, executable=True, error=None):
        instance = super().__new__(cls, value)
        instance.rows = rows
        instance.executable = executable
        instance.error = error
        return instance


def query_db(cypher_string, dsgdb_conf: HeraclesDsgInterface = None):
    if dsgdb_conf is None:
        raise ValueError(
            "query_db called with dsgdb_conf=None. Did you forget to bind the config to the tool?"
        )
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
            v = db.query("MATCH (n: Object) RETURN COUNT(*) as count")
            count = v[0]["count"]
            assert count == dsgdb_conf.n_object_verification, (
                f"Connected database has {count} objects ({dsgdb_conf.n_object_verification} expected)"
            )
        try:
            rows = db.query(cypher_string)
            return CypherQueryResult(str(rows), rows=rows, executable=True)
        except Exception as ex:
            # Keep returning the error to the agent so it can repair its query, while
            # retaining a machine-readable failure flag for benchmark validation.
            logger.debug("Cypher query failed: %s", ex)
            return CypherQueryResult(
                str(ex), rows=None, executable=False, error=f"{type(ex).__name__}: {ex}"
            )


def bind_query_db(dsgdb_conf: HeraclesDsgInterface):
    def run_cypher_query(cypher_string):
        return query_db(cypher_string, dsgdb_conf=dsgdb_conf)

    return run_cypher_query


cypher_tool = ToolDescription(
    name="run_cypher_query",
    description="An interface for running Cypher queries on a Neo4j database containing a 3D Scene Graph.",
    parameters=[
        FunctionParameter("cypher_string", str, "Your Cypher query"),
    ],
    function=query_db,
)

register_tool(cypher_tool)
logger.debug(f"Registered tools: {ToolRegistry.registered_tool_summary()}")
