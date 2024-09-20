# Metadata schemas for JHU Turbulence Database

Files in this folder
* **jhtdb_schema.py**: script using pydantic data model to create JSON schema files for main data model and for classes providing annotations to that model for server and client-side use.
  * **jhtdb-schema.json**: JSON schema generated from pydantic main data model 
  * **jhtdb-server-schema.json**: JSON schema generated from pydantic server-side data model 
  * **jhtdb-schema.json**: JSON schema generated from pydantic client-side data model for cutout service
  * **jhtdb-schema.json**: JSON schema generated from pydantic client-side data model for point queries

**jhtdb_schema.py** defines a data model for JHTDB turbelence metadata.
It does so using ppydantic classes.
These can be exported as JSON schema files, 4 of which are also included here.