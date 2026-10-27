"""Every action, with its metadata, in one append-only log.

The agent's tools are thin wrappers that log each call. The lab writes hidden `sim`
events recording what really happened. Human interventions - what was asked, what the
technician reported, how long it took - are logged too. Scoring reads only this log.
"""
