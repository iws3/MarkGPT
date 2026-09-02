from langchain_core.tools import tool



@tool
def get_module_deadline(module_name:str)->str:
    """
    Look up the submission deadline for  a name SEED bootcamp module
    
    Args:
    module_name: The module name, e.g, 'ANN', 'CNN', or RNN
    """
    deadlines={
        "ANN":"2026-09-01",
        "CNN":"2026-09-25",
        "RNN":"2026-09-29"
    }
    return deadlines.get(module_name, "No deadline found for that module.")

# A tool can be called directly
print(get_module_deadline.invoke({"module_name":"CNN"})) 
# --> [output + prompt --> result]
@tool
def count_students_in_module(module_name:str)->str:
    """Look up how many students are enrolled in a named bootcamp module"""
    counts={"ANN":23, "CNN": 34, "RNN":45}
    return str(counts.get(module_name, 0))
