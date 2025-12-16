from main import SIRsimulation

if __name__ == "__main__":
    sim = SIRsimulation(
        gridsize=(100,100),
        
    )

    sim.animate()