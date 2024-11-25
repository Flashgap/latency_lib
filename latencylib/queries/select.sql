SELECT
    samples.trace_id as trace_id,
    span.span_id as span_id,
    span.name as span_name,
    parent.span_id as parent_span_id,
    parent.name as parent_name,
    root.name as root_name,
    root.start_time as root_start_time,
    root.span_id as root_span_id,
    STRING(json_extract(root.attributes, "$['http.method']")) as method,
    span.start_time as start_time,
    span.duration_nano/1000000 as duration_ms
FROM (
    SELECT trace_id FROM (
        SELECT DISTINCT(trace_id) FROM traces.spans s
        WHERE contains_substr(name, @name) AND s.start_time BETWEEN @start_time AND @end_time
    )
  WHERE RAND()<=@sample_rate
) AS samples
INNER JOIN (
    SELECT * FROM `traces.spans`
    WHERE contains_substr(name, @name)
) AS root
ON samples.trace_id = root.trace_id
INNER JOIN `traces.spans` span
ON root.trace_id = span.trace_id
LEFT JOIN (
  SELECT * FROM `traces.spans`
) parent
ON parent.span_id = span.parent_span_id
ORDER BY span.start_time
