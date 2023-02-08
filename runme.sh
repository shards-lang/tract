#!/bin/sh
	
# RUSTFLAGS=-Awarnings cargo +1.65.0 test --profile opt-no-lto -p tract-core -- --nocapture --test-threads 1 monterey

 set -ex

RUST_VERSION=${RUST_VERSION:-1.65.0}

rustup toolchain install $RUST_VERSION
RUSTFLAGS=-Awarnings cargo +$RUST_VERSION test -p tract-data

for i in `seq 1 100` ; do
	echo $i
	RUSTFLAGS=-Awarnings cargo +$RUST_VERSION test --profile opt-no-lto -p tract-data -- --test-threads 1 crasher_monterey
	#RUSTFLAGS=-Awarnings cargo +1.65.0 test --profile opt-no-lto -p tract-core -- --nocapture --test-threads 1 kali
#	valgrind ./target/opt-no-lto/deps/tract_core-dc1e23fe486abe6d --test-threads 1 --nocapture ops::cnn::conv::proptest::crasher_monterey
 done
