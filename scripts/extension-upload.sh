#!/bin/bash

# Extension upload script

# Usage: ./extension-upload.sh <name> <extension_version> <duckdb_version> <architecture> <s3_bucket> <copy_to_latest> <copy_to_versioned>
# <name>                : Name of the extension
# <extension_version>   : Version (commit / version tag) of the extension
# <duckdb_version>      : Version (commit / version tag) of DuckDB
# <architecture>        : Architecture target of the extension binary
# <s3_bucket>           : S3 bucket to upload to
# <copy_to_latest>      : Set this as the latest version ("true" / "false", default: "false")
# <copy_to_versioned>   : Set this as a versioned version that will prevent its deletion

set -e

if [[ $4 == wasm* ]]; then
  ext="/tmp/extension/$1.duckdb_extension.wasm"
else
  ext="/tmp/extension/$1.duckdb_extension"
fi

echo $ext

script_dir="$(dirname "$(readlink -f "$0")")"

# The build artifact already ends with a 256-byte signature placeholder
# (zero-filled when unsigned). Strip it before computing the hash and
# re-appending a fresh signature — otherwise we'd end up with 512 bytes
# of trailing data and the metadata footer would no longer be at the
# fixed offset DuckDB v1.5.3+ checks, causing the install to fail with
# "metadata at the end of the file is invalid".
#
# wasm artifacts from current extension-ci-tools also end with a complete
# `duckdb_signature` custom section (metadata + zeroed 256-byte placeholder).
# Appending a second section breaks LOAD in DuckDB-Wasm ("Unknown ABI type",
# #192), so a wasm artifact that already has the section is treated like a
# native one: only its placeholder is replaced. Older wasm artifacts without
# the section still get one appended.
has_wasm_sig_section=0
if [[ $4 == wasm* ]] && tail -c 1024 $ext | grep -qa 'duckdb_signature'; then
  has_wasm_sig_section=1
fi

cat $ext > $ext.append
if [[ $4 != wasm* || $has_wasm_sig_section == 1 ]]; then
  truncate -s -256 $ext.append
fi

if [[ $4 == wasm* && $has_wasm_sig_section == 0 ]]; then
  # 0 for custom section
  # 113 in hex = 275 in decimal, total lenght of what follows (1 + 16 + 2 + 256)
  # [1(continuation) + 0010011(payload) = \x93, 0(continuation) + 10(payload) = \x02]
  echo -n -e '\x00' >> $ext.append
  echo -n -e '\x93\x02' >> $ext.append
  # 10 in hex = 16 in decimal, lenght of name, 1 byte
  echo -n -e '\x10' >> $ext.append
  echo -n -e 'duckdb_signature' >> $ext.append
  # the name of the WebAssembly custom section, 16 bytes
  # 100 in hex, 256 in decimal
  # [1(continuation) + 0000000(payload) = ff, 0(continuation) + 10(payload)],
  # for a grand total of 2 bytes
  echo -n -e '\x80\x02' >> $ext.append
fi

# (Optionally) Sign binary
if [ "$DUCKDB_EXTENSION_SIGNING_PK" != "" ]; then
  echo "$DUCKDB_EXTENSION_SIGNING_PK" > private.pem
  $script_dir/../duckdb/scripts/compute-extension-hash.sh $ext.append > $ext.hash
  openssl pkeyutl -sign -in $ext.hash -inkey private.pem -pkeyopt digest:sha256 -out $ext.sign
  rm -f private.pem
fi

# Signature is always there, potentially defaulting to 256 zeros
truncate -s 256 $ext.sign

# append signature to extension binary
cat $ext.sign >> $ext.append

# A wasm extension must carry exactly one duckdb_signature section (#192).
if [[ $4 == wasm* ]]; then
  n_sig=$(tail -c 2048 $ext.append | grep -ao 'duckdb_signature' | wc -l)
  if [[ $n_sig != 1 ]]; then
    echo "error: $ext.append has $n_sig duckdb_signature sections, expected 1" >&2
    exit 1
  fi
fi

# compress extension binary
if [[ $4 == wasm_* ]]; then
  brotli < $ext.append > "$ext.compressed"
else
  gzip < $ext.append > "$ext.compressed"
fi

set -e

# Abort if AWS key is not set
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "No AWS key found, skipping.."
    exit 0
fi

# upload versioned version
if [[ $7 = 'true' ]]; then
  if [[ $4 == wasm* ]]; then
    aws s3 cp $ext.compressed s3://$5/$1/$2/$3/$4/$1.duckdb_extension.wasm --acl public-read --content-encoding br --content-type="application/wasm"
  else
    aws s3 cp $ext.compressed s3://$5/$1/$2/$3/$4/$1.duckdb_extension.gz --acl public-read
  fi
fi

# upload to latest version
if [[ $6 = 'true' ]]; then
  if [[ $4 == wasm* ]]; then
    aws s3 cp $ext.compressed s3://$5/$3/$4/$1.duckdb_extension.wasm --acl public-read --content-encoding br --content-type="application/wasm"
  else
    aws s3 cp $ext.compressed s3://$5/$3/$4/$1.duckdb_extension.gz --acl public-read
  fi
fi
