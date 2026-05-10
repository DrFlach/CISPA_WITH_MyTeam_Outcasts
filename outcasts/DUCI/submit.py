import os
import sys
import requests

BASE_URL = "http://35.192.205.84"
API_KEY = "31fd8c57481049a79ce9e526df488d56"
TASK_ID = "11-duci"

FILE_PATH = "output/submission.csv"


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def submit():

    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    print(f"Submitting: {FILE_PATH}")

    try:

        with open(FILE_PATH, "rb") as f:

            files = {
                "file": (
                    os.path.basename(FILE_PATH),
                    f,
                    "text/csv",
                ),
            }

            response = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={
                    "X-API-Key": API_KEY
                },
                files=files,
                timeout=(30, 300),
            )

        print("\nHTTP STATUS:", response.status_code)

        try:
            body = response.json()
        except Exception:
            body = response.text

        print("\nSERVER RESPONSE:")
        print(body)

        response.raise_for_status()

        print("\nSUCCESSFULLY SUBMITTED")

    except requests.exceptions.RequestException as e:

        print("\nSUBMISSION FAILED")
        print(e)

        if hasattr(e, "response") and e.response is not None:

            try:
                print(e.response.json())
            except Exception:
                print(e.response.text)

        sys.exit(1)


if __name__ == "__main__":
    submit()