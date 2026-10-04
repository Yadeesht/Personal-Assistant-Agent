import sys
import os
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="JARVIS Sacrificial Code Sandbox Server")

# Ensure sandbox directory exists
SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"
SANDBOX_DIR.mkdir(exist_ok=True)

class CodeExecutionRequest(BaseModel):
    code: str

@app.post("/execute")
async def execute_code(req: CodeExecutionRequest):
    # Create a unique temp file in the sandbox folder
    file_id = uuid.uuid4().hex
    file_path = SANDBOX_DIR / f"exec_{file_id}.py"
    
    try:
        # Write the code
        file_path.write_text(req.code, encoding="utf-8")
        
        # Spawn python subprocess
        # Set cwd to sandbox directory so any file output stays local
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(SANDBOX_DIR)
        )
        
        try:
            # Enforce 30-second timeout
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            exit_code = process.returncode
            
            return {
                "status": "success" if exit_code == 0 else "error",
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": exit_code
            }
            
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            return {
                "status": "error",
                "stdout": "",
                "stderr": "Execution Timeout: Python process took longer than 30 seconds to complete.",
                "exit_code": -1
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sandbox execution setup error: {str(e)}")
        
    finally:
        # Clean up temp file
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
