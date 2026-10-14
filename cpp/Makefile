CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -march=native -Wall -Wextra
all: pulsar_test pulsar_core_v1_test pulsar_speculative_v2_test pulsar_tree_v3_test pulsar_stage4_bench pulsar_stage4_bench_test pulsar_stage5_native
pulsar_test: pulsar_test.cpp pulsar_engine.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_test.cpp
pulsar_core_v1_test: pulsar_core_v1_test.cpp pulsar_core_v1.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_core_v1_test.cpp
pulsar_speculative_v2_test: pulsar_speculative_v2_test.cpp pulsar_speculative_v2.hpp pulsar_core_v1.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_speculative_v2_test.cpp
pulsar_tree_v3_test: pulsar_tree_v3_test.cpp pulsar_tree_v3.hpp pulsar_speculative_v2.hpp pulsar_core_v1.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_tree_v3_test.cpp
pulsar_stage4_bench: pulsar_stage4_bench.cpp pulsar_stage4_bench.hpp pulsar_tree_v3.hpp pulsar_engine.hpp pulsar_speculative_v2.hpp pulsar_core_v1.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_stage4_bench.cpp
pulsar_stage4_bench_test: pulsar_stage4_bench_test.cpp pulsar_stage4_bench.hpp pulsar_tree_v3.hpp pulsar_core_v1.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_stage4_bench_test.cpp
pulsar_stage5_native: pulsar_stage5_native.cpp pulsar_stage4_bench.hpp pulsar_tree_v3.hpp pulsar_engine.hpp pulsar_core_v1.hpp
	$(CXX) $(CXXFLAGS) -o $@ pulsar_stage5_native.cpp
stage5_test:
	cd .. && python3 verification/experiments/test_stage5.py
stage5_native_test: pulsar_stage5_native
	./pulsar_stage5_native --model inputs/model.bin --self-test
stage6_transformers_test:
	cd .. && python3 verification/experiments/test_stage6_transformers.py
stage6_transformers:
	cd .. && python3 verification/experiments/run_stage6_transformers.py
test: pulsar_test pulsar_core_v1_test pulsar_speculative_v2_test pulsar_tree_v3_test pulsar_stage4_bench_test stage5_test stage5_native_test
	./pulsar_core_v1_test
	./pulsar_speculative_v2_test
	./pulsar_tree_v3_test
	./pulsar_stage4_bench_test
	cd inputs/.. && ./pulsar_test inputs/model.bin
clean:
	rm -f pulsar_test pulsar_core_v1_test pulsar_speculative_v2_test pulsar_tree_v3_test pulsar_stage4_bench pulsar_stage4_bench_test pulsar_stage5_native
