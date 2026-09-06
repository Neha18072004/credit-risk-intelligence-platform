"""SQL validation: the hallucination and injection control for talk-to-data.

An LLM writes the SQL, so the SQL is untrusted input. This module is the layer
that makes that acceptable. Its guiding rule is that **a wrongly rejected safe
query is always better than an executed unsafe one**, so every check fails
closed.

The single most important design decision here: the validated statement is
**re-serialised from the parsed syntax tree**, never passed through as the
string the model produced. Anything the parser did not understand and represent
-- a trailing stacked statement, stray punctuation -- cannot survive that round
trip, because it is simply not in the tree that gets printed back out. Comments
are the exception worth naming: sqlglot *does* retain them on the nodes and
re-emits them, so they are stripped explicitly rather than assumed away.

The layered defences, in order:

1. Fence stripping, comment stripping and single-statement enforcement.
2. Structural check: the root must be a ``SELECT`` (optionally behind a ``WITH``).
3. Node blacklist: any write, DDL or command node anywhere in the tree is fatal.
4. Function blacklist: file, network and sleep primitives.
5. Schema whitelist: every table and column must exist in the live database.
6. Row cap: a ``LIMIT`` is injected or tightened.
7. Re-serialisation from the tree.

A read-only database role and a server-side statement timeout sit behind all of
this, so a bypass of any single layer is still not sufficient to cause harm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

import sqlglot
from sqlglot import exp

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expression types that must never appear anywhere in a generated statement.
FORBIDDEN_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Command, exp.Transaction, exp.Commit,
    exp.Rollback, exp.Use, exp.Set, exp.AlterColumn, exp.Merge,
)

# Functions that read files, reach the network, sleep, or escalate privilege.
FORBIDDEN_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
        "pg_sleep", "pg_sleep_for", "pg_sleep_until", "sleep",
        "lo_import", "lo_export", "dblink", "dblink_exec", "dblink_connect",
        "copy", "system", "shell", "exec", "execute", "eval",
        "load_extension", "readfile", "writefile", "load_file",
        "current_setting", "set_config", "pg_terminate_backend", "pg_cancel_backend",
        "query_to_xml", "xmlparse", "pg_client_encoding", "version",
    }
)

# Keyword-level backstop. The parser is the real defence; this catches text that
# fails to parse cleanly and would otherwise reach the "unparseable" branch with
# something obviously hostile in it.
SUSPICIOUS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(drop|delete|truncate|insert|update|alter|grant|revoke|create)\s+"
    r"(table|database|schema|index|view|role|user|from|into)\b",
    re.IGNORECASE,
)

_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*```(?:sql)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE
)


def _strip_comments(tree: exp.Expression) -> None:
    """Remove every comment attached anywhere in the parse tree, in place."""
    for node in tree.walk():
        if getattr(node, "comments", None):
            node.comments = None


@dataclass
class ValidationResult:
    """Outcome of validating one generated statement."""

    is_valid: bool
    sql: str = ""
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    limit_applied: int | None = None

    def __bool__(self) -> bool:
        return self.is_valid


class SQLValidator:
    """Validates and rewrites LLM-generated SQL against a live schema.

    Args:
        schema: Mapping of table name to its set of column names, lower-cased.
            Sourced from the real database, so a renamed column immediately
            becomes a rejection rather than a runtime error.
        max_rows: Hard row cap injected as a ``LIMIT``.
        dialect: sqlglot dialect used for both parsing and re-serialisation.
    """

    def __init__(
        self,
        schema: dict[str, set[str]],
        max_rows: int | None = None,
        dialect: str = "postgres",
    ) -> None:
        self.schema = {table.lower(): {c.lower() for c in columns} for table, columns in schema.items()}
        self.max_rows = max_rows or settings.sql_max_rows
        self.dialect = dialect

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def strip_fences(sql: str) -> str:
        """Remove markdown code fences and trailing semicolons.

        Models wrap SQL in ``` blocks regardless of instructions, so this is
        normalisation rather than a security control.
        """
        match = _FENCE_PATTERN.match(sql.strip())
        cleaned = match.group(1) if match else sql.strip()
        return cleaned.strip().rstrip(";").strip()

    def _reject(self, reason: str, **extra: object) -> ValidationResult:
        """Build a rejection, logging it so refusals are auditable."""
        logger.warning("SQL rejected: %s", reason)
        return ValidationResult(is_valid=False, reason=reason, **extra)  # type: ignore[arg-type]

    # ---------------------------------------------------------- validation --
    def validate(self, sql: str) -> ValidationResult:
        """Validate a statement and return a safe, rewritten version.

        Args:
            sql: Raw text produced by the language model.

        Returns:
            A :class:`ValidationResult`. When ``is_valid`` is True, ``sql`` holds
            the statement that is safe to execute -- re-serialised from the
            parse tree with a row cap applied. When False, ``reason`` explains
            the refusal in terms a user can act on.
        """
        if not sql or not sql.strip():
            return self._reject("Empty query.")

        cleaned = self.strip_fences(sql)
        if not cleaned:
            return self._reject("Query contained no SQL after removing formatting.")

        # --- 1. exactly one statement ---
        try:
            statements = sqlglot.parse(cleaned, read=self.dialect)
        except Exception as error:  # noqa: BLE001 - any parse failure is a rejection
            return self._reject(f"Could not parse the SQL: {error}")

        statements = [statement for statement in statements if statement is not None]
        if not statements:
            return self._reject("No executable statement found.")
        if len(statements) > 1:
            return self._reject(
                f"Only one statement is allowed; {len(statements)} were provided. "
                "Stacked statements are always rejected."
            )

        tree = statements[0]

        # --- 2. must be a SELECT ---
        if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
            if isinstance(tree, exp.With) or tree.args.get("with"):
                pass  # a CTE whose body is checked below
            else:
                return self._reject(
                    f"Only SELECT queries are permitted; received a "
                    f"{type(tree).__name__.upper()} statement."
                )

        if settings.sql_allow_only_select and not self._resolves_to_select(tree):
            return self._reject("Only SELECT queries are permitted.")

        # --- 3. forbidden node types anywhere in the tree ---
        for node_type in FORBIDDEN_NODES:
            found = list(tree.find_all(node_type))
            if found:
                return self._reject(
                    f"Statement contains a forbidden {node_type.__name__.upper()} operation. "
                    "This interface is strictly read-only."
                )

        # --- 4. forbidden functions ---
        for function in tree.find_all(exp.Anonymous, exp.Func):
            name = (getattr(function, "name", "") or "").lower()
            if name in FORBIDDEN_FUNCTIONS:
                return self._reject(f"Function {name}() is not permitted.")

        # --- 5. keyword backstop ---
        if SUSPICIOUS_PATTERN.search(cleaned):
            return self._reject("Statement contains a data-modifying keyword pattern.")

        # --- 6. schema whitelist ---
        schema_check = self._check_schema(tree)
        if not schema_check.is_valid:
            return schema_check

        # --- 7. strip comments, apply the row cap, re-serialise from the tree ---
        # sqlglot attaches comments to nodes and re-emits them, so a round trip
        # alone does not remove them. They are inert (stacked statements are
        # already rejected, so a comment cannot smuggle execution), but stripping
        # them keeps the executed text free of any model-authored payload and
        # makes the audit log show exactly the logic that ran.
        _strip_comments(tree)
        tree, limit_applied = self._apply_limit(tree)
        try:
            safe_sql = tree.sql(dialect=self.dialect, pretty=True)
        except Exception as error:  # noqa: BLE001
            return self._reject(f"Could not safely rewrite the query: {error}")

        return ValidationResult(
            is_valid=True,
            sql=safe_sql,
            tables=schema_check.tables,
            columns=schema_check.columns,
            warnings=schema_check.warnings,
            limit_applied=limit_applied,
        )

    # ------------------------------------------------------------ internals --
    @staticmethod
    def _resolves_to_select(tree: exp.Expression) -> bool:
        """True when the statement ultimately produces rows via SELECT."""
        if isinstance(tree, (exp.Select, exp.Union, exp.Subquery)):
            return True
        if isinstance(tree, exp.With):
            return isinstance(tree.this, (exp.Select, exp.Union))
        return False

    def _check_schema(self, tree: exp.Expression) -> ValidationResult:
        """Verify every referenced table and column exists in the live schema.

        This is the anti-hallucination check proper. A model that invents
        ``applications.credit_score`` fails here with a message naming the real
        columns, which is both safer and more useful than a database error.
        """
        # CTE names are legitimate table references that do not exist in the DB.
        cte_names = {
            cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name
        }

        # A CTE that shadows a real table is refused. Testing confirmed it is
        # not an escape -- the CTE body is still fully validated, so a shadowed
        # name cannot smuggle in a catalog table, an unknown column or a write.
        # It is rejected anyway because it makes a statement mean something
        # different from what it appears to say, no legitimate generated query
        # needs it, and this layer fails closed by policy.
        shadowed = cte_names & set(self.schema)
        if shadowed:
            return self._reject(
                f"A CTE may not reuse the name of a real table: {', '.join(sorted(shadowed))}. "
                "Rename the CTE."
            )

        referenced: set[str] = set()
        alias_map: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            name = (table.name or "").lower()
            if not name or name in cte_names:
                continue
            referenced.add(name)
            alias = (table.alias or "").lower()
            if alias:
                alias_map[alias] = name
            alias_map[name] = name

        unknown_tables = sorted(referenced - set(self.schema))
        if unknown_tables:
            return self._reject(
                f"Unknown table(s): {', '.join(unknown_tables)}. "
                f"Available tables are: {', '.join(sorted(self.schema))}.",
                tables=sorted(referenced),
            )

        if not referenced and not cte_names:
            return self._reject("Query does not read from any known table.")

        # Column names that are produced by the query itself are legitimate
        # references later in the statement (ORDER BY on an alias, for example).
        produced: set[str] = {
            alias.alias_or_name.lower()
            for alias in tree.find_all(exp.Alias)
            if alias.alias_or_name
        } | cte_names

        allowed_columns: set[str] = set()
        for table in referenced:
            allowed_columns |= self.schema[table]

        unknown_columns: set[str] = set()
        seen_columns: set[str] = set()
        for column in tree.find_all(exp.Column):
            name = (column.name or "").lower()
            if not name or name == "*":
                continue
            seen_columns.add(name)

            qualifier = (column.table or "").lower()
            if qualifier and qualifier in alias_map:
                table = alias_map[qualifier]
                if name not in self.schema[table] and name not in produced:
                    unknown_columns.add(f"{qualifier}.{name}")
            elif name not in allowed_columns and name not in produced:
                unknown_columns.add(name)

        if unknown_columns:
            hint = ", ".join(sorted(allowed_columns)[:12])
            return self._reject(
                f"Unknown column(s): {', '.join(sorted(unknown_columns))}. "
                f"Columns available on the referenced tables include: {hint}...",
                tables=sorted(referenced),
                columns=sorted(seen_columns),
            )

        warnings: list[str] = []
        if any(isinstance(node, exp.Star) for node in tree.find_all(exp.Star)):
            warnings.append("SELECT * used; consider naming columns explicitly.")

        return ValidationResult(
            is_valid=True, tables=sorted(referenced), columns=sorted(seen_columns), warnings=warnings
        )

    def _apply_limit(self, tree: exp.Expression) -> tuple[exp.Expression, int | None]:
        """Inject or tighten a ``LIMIT`` so no query can return unbounded rows.

        An aggregate with no GROUP BY returns a single row, so a limit there is
        harmless; it is applied uniformly rather than special-cased, because a
        uniform rule is easier to reason about than a clever one.
        """
        target = tree.this if isinstance(tree, exp.With) else tree
        if not isinstance(target, (exp.Select, exp.Union)):
            return tree, None

        existing = target.args.get("limit")
        if existing is not None:
            try:
                current = int(existing.expression.this)
            except (AttributeError, TypeError, ValueError):
                current = None
            if current is not None and current <= self.max_rows:
                return tree, current

        limited = target.limit(self.max_rows)
        if isinstance(tree, exp.With):
            tree.set("this", limited)
            return tree, self.max_rows
        return limited, self.max_rows


def build_validator(schema: dict[str, set[str]] | None = None) -> SQLValidator:
    """Construct a validator against the live database schema.

    Args:
        schema: Override the schema; read from the database when omitted.

    Returns:
        A configured :class:`SQLValidator`.
    """
    if schema is None:
        from src.talk_to_data.query_runner import fetch_schema

        schema = fetch_schema()
    return SQLValidator(schema=schema, max_rows=settings.sql_max_rows)
