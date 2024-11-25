# This query selects specific spans from the traces table related to Datastore entities.
SELECT *
FROM traces.spans
WHERE contains_substr(name, @name) AND s.start_time BETWEEN @start_time AND @end_time
AND contains_substr(JSON_VALUE_ARRAY(json_extract(attributes, "$['datastore.entities']")), @entity_name)
LIMIT @limit
