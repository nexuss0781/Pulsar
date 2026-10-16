#include <pulsar/batch_generation.hpp>
#include <pulsar/speculative_generation.hpp>
#include <pulsar/tree_generation.hpp>

#include <string>
#include <vector>

int main() {
    using namespace pulsar::runtime;
    SamplingConfig sampling;
    sampling.temperature = 0.0;
    sampling.top_k = 0;
    sampling.top_p = 1.0;
    const Probabilities law = make_distribution({1.0F, 2.0F}, {}, sampling);
    return law.size() == 2 && law[1] == 1.0 &&
                   std::string(to_string(FinishReason::cancelled)) == "cancelled"
               ? 0
               : 1;
}
