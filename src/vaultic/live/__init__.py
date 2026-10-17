"""Test Lab live scoring (product layer, D182-): the frozen stage-2 system scoring one
transaction at a time with every feature computed online, point-in-time, from the stored history
strictly before the transaction. Nothing here changes a frozen research result, config or freeze
record; parity with the offline features and the stored final-run scores is checked before the
Test Lab serves anything (live/parity.py)."""
