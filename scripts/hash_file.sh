file="$1"

if [[ -z "$file" ]]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

if [[ ! -f "$file" ]]; then
  echo "File not found: $file"
  exit 1
fi

sha256sum "$file"