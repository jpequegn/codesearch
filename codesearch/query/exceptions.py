class QueryError(Exception):
    """Raised for database/LanceDB errors during search."""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class FilterError(Exception):
    """Raised for invalid filter specifications."""

    def __init__(self, message: str):
        super().__init__(message)
