
#ifndef SIMULATION_CONFIG_HPP
#define SIMULATION_CONFIG_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <fstream>


// Struct for Neighborhood definition
struct Neighborhood {
    std::string type;
    int range;
};

// Struct for Simulation Details
struct SimulationDetails {
    std::string name;
    int imageSelectionIndex;
};


// Struct for Cell Configuration
struct CellConfig {
    std::string delay;
    std::string model;
    std::vector<Neighborhood> neighborhood;
};

// Struct for the Scenario
struct Scenario {
    std::vector<int> shape;
    std::vector<int> origin;
    bool wrapped;
    CellConfig defaultCell;
    SimulationDetails simulationDetails;
    std::string simulationType;
};


class JSON_LOADER{
    public:
        JSON_LOADER() = default;
        // Function to load the scenario from a JSON file
        Scenario loadScenario(const std::string& filepath) {
            std::ifstream inFile(filepath);
            nlohmann::json j;
            inFile >> j;
        
            Scenario scenario;
        
            // Load scenario
            scenario.shape = j["scenario"]["shape"].get<std::vector<int>>();
            scenario.origin = j["scenario"]["origin"].get<std::vector<int>>();
            scenario.wrapped = j["scenario"]["wrapped"];
        
            // Load simulation details
            scenario.simulationDetails.name = j["simulation_details"]["name"];
            scenario.simulationDetails.imageSelectionIndex = j["simulation_details"]["image_selection"];
        
            // Load cells section 
            const auto& defaultCell = j["cells"]["default"];
            scenario.defaultCell.delay = defaultCell["delay"];
            scenario.defaultCell.model = defaultCell["model"];
        
            // Load neighborhood info
            for (const auto& neigh : defaultCell["neighborhood"]) {
                Neighborhood n;
                n.type = neigh["type"];
                n.range = neigh["range"];
                scenario.defaultCell.neighborhood.push_back(n);
            }
        
            scenario.simulationType = j["simulation_type"]; 
        
            return scenario;
        }        
};

#endif // SIMULATION_CONFIG_HPP

