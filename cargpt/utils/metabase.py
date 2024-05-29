import os

import requests
from dotenv import load_dotenv

_ = load_dotenv()


# TODO: move to yaak-datasets

DB_MAP = {"example": 1, "stage": 2, "prod": 3}


class MetabaseCaller:
    def __init__(self) -> None:
        self.host = "https://metabase.yaak.dev"
        super().__init__()

    def request_incidents(
        self, drive_id: str, database_id: int = DB_MAP["prod"]
    ) -> list[str]:
        headers = self._build_headers()
        # build it here not to throw error when not using the class
        if "/" in drive_id:
            drive_id = drive_id.split("/")[1]
        query = self._build_incident_request_quiery(drive_id)
        data = {
            "database": database_id,  # Replace 1 with the ID of the database containing the sample dataset
            "type": "native",
            "native": {"query": query},
        }

        # NOTE: may be useful to wrap with multiple exceptions try/except
        response = requests.post(
            f"{self.host}/api/dataset", headers=headers, json=data, timeout=10
        )
        if not response.ok:
            pass

        return response.json()["data"]["rows"]

    @staticmethod
    def _build_headers():
        if not os.getenv("METABASE_API"):
            msg = "METABASE_API is not set"
            raise ValueError(msg)
        if not os.getenv("CF_ACCESS_CLIENT_SECRET"):
            msg = "CF_ACCESS_CLIENT_SECRET is not set"
            raise ValueError(msg)
        if not os.getenv("CF_ACCESS_CLIENT_ID"):
            msg = "CF_ACCESS_CLIENT_ID is not set"
            raise ValueError(msg)
        return {
            "Content-Type": "application/json",
            "X-API-KEY": os.getenv("METABASE_API"),
            "CF-Access-Client-Secret": os.getenv("CF_ACCESS_CLIENT_SECRET"),
            "CF-Access-Client-Id": os.getenv("CF_ACCESS_CLIENT_ID"),
        }

    @staticmethod
    def _build_incident_request_quiery(drive_id: str):
        return f"""
            SELECT
              annotations.end_timestamp AS annotation_end,
              Tags.name AS tag_name
            FROM
              annotations
              LEFT JOIN public.tags Tags ON public.annotations.tag_id = Tags.id
              LEFT JOIN public.sessions Session ON public.annotations.session_id = Session.id
            WHERE
              Session.canonical_name IN (
                '{drive_id}'
              )
            ORDER BY
              annotations.end_timestamp ASC
          """


metabase = MetabaseCaller()
