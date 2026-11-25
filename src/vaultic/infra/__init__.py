"""Platform infrastructure for the Vaultic demo (Sprint 3, D230, D231): database connection and
migrations, and later auth, cache and streaming. Named 'infra', not 'platform': a package called
platform shadows Python's stdlib module when src/vaultic is on sys.path (the reason for D2).

Nothing here feeds a model, a metric or a research table; the no-Docker demo does not use it."""
