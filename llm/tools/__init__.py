# from .semantic_memory import SemanticMemory

# Keep package import resilient: some workspaces expose `WorkingMemory` not `WorkingMemoryer`.
try:
	from .working_memoryer import WorkingMemoryer
except Exception:
	try:
		from .working_memoryer import WorkingMemory as WorkingMemoryer
	except Exception:
		WorkingMemoryer = None