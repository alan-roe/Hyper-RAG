# Hyper-RAG Development Tools

This directory contains development tools for the Hyper-RAG project.

## find_and_fix_chinese.py

A tool to find and analyze Chinese characters in the codebase that need localization.

### Features

- Scans both backend (Python) and frontend (TypeScript/React) code
- Identifies Chinese text that needs localization
- Categorizes findings into:
  - 🔴 Needs localization (hardcoded strings in UI/messages)
  - 🟡 Possibly already using i18n (strings near translation functions)
  - 🟢 In comments/debug (safe to ignore)
- Generates detailed reports with file locations and context
- Provides fix suggestions for both backend and frontend

### Usage

```bash
# Scan all code (backend and frontend)
uv run python tools/find_and_fix_chinese.py

# Scan only backend
uv run python tools/find_and_fix_chinese.py --target backend

# Scan only frontend
uv run python tools/find_and_fix_chinese.py --target frontend

# Save detailed results to JSON
uv run python tools/find_and_fix_chinese.py --output results.json

# Show fix suggestions
uv run python tools/find_and_fix_chinese.py --fix

# Scan a custom path
uv run python tools/find_and_fix_chinese.py --path /custom/path
```

### Options

- `--target`: Choose what to scan (`backend`, `frontend`, or `all`)
- `--output`: Save detailed results to a JSON file
- `--fix`: Show suggestions for fixing the Chinese strings
- `--path`: Custom path to scan (default: auto-detect web-ui directory)

### Output Format

The tool generates:
1. Console report with categorized findings
2. Optional JSON file with detailed results including:
   - Summary statistics
   - Full list of strings needing localization
   - File locations and line numbers
   - Context for each occurrence

### Example Output

```
================================================================================
CHINESE CHARACTER ANALYSIS REPORT - BACKEND
================================================================================

Found 401 unique Chinese text strings in 7 files
================================================================================

🔴 NEEDS LOCALIZATION:
----------------------------------------
  Text: '文件不存在'
  Found in 6 location(s):
    - backend/main.py:821
      Context: error_msg = f"文件不存在: {file_id}"
    ...

🟡 POSSIBLY ALREADY USING I18N:
----------------------------------------
  Text: '成功'
  Found in 4 location(s):
    - backend/main.py:774
      Context: print(f"文件上传完成，成功: {successful}/{total}")
    ...

🟢 IN COMMENTS/DEBUG (probably OK):
----------------------------------------
  Text: '获取文件信息'
  Example: backend/main.py:817
    ...

================================================================================
SUMMARY:
  - Needs localization: 117 unique strings
  - Possibly already localized: 60 unique strings
  - In comments/debug: 224 unique strings
================================================================================
```

### Integration with i18n Systems

#### Backend (Python)
The tool suggests using the `translations.py` module with the `t()` function:
```python
from translations import t

# Replace
error_msg = "文件不存在"
# With
error_msg = t('file_not_exist')
```

#### Frontend (React/TypeScript)
The tool suggests using `react-i18next`:
```tsx
import { useTranslation } from 'react-i18next';

// Replace
<div>文件不存在</div>
// With
<div>{t('file_not_exist')}</div>
```

### Files Scanned

- **Backend**: `.py`, `.json`, `.yaml`, `.yml`
- **Frontend**: `.tsx`, `.ts`, `.jsx`, `.js`, `.css`, `.scss`
- **Common**: `.md`, `.txt` (when target is 'all')

### Excluded Directories

The tool automatically skips:
- `__pycache__`, `.git`, `node_modules`
- `.venv`, `venv`, `dist`, `build`
- `.next`, `coverage`, `.pytest_cache`
- `tools` (to avoid scanning itself)
- Files with 'translations' or 'i18n' in the name