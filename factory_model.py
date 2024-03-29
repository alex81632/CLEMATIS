import sys
import argparse
import numpy as np
import pandas as pd
from igraph import *
#from model_gen import ModelGenerator
from model_gen import ModelGeneratorNS
from model_gen import DynamicManufacturing
# Para rodar basta inserir na linha de comando:
# python factory_model.py -n 20 -s 5 -r 2 -o output.txt


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--nodes", required=True, type=int)
ap.add_argument("-s", "--production_steps", required=True, type=int)
ap.add_argument("-f", "--first_step", type=int, default=-1)
ap.add_argument("-l", "--last_step", type=int, default=-1)
ap.add_argument("-r", "--seed", required=True, type=int) 
ap.add_argument("-o", "--output", required=True, default="-")
ap.add_argument("-t", "--production", default="constant")
ap.add_argument("-i", "--samples", type=int, default=30)

args = vars(ap.parse_args())

print("[INFO] Generating graph...")

mg = ModelGeneratorNS(n=args["nodes"],
                      s=args["production_steps"],
                      first_step=args["first_step"],
                      last_step=args["last_step"],
					  buffer_size=3,
                      rng =np.random.default_rng( args["seed"]))
#rng= args["rng"]
#ws, edges, edge_attr, vertex_attr = mg.generate_graph()
ws, edges, vertex_attr = mg.generate_graph()
# print(f"ws: {ws}")
# print(f"edges: {edges}")
# print(f"vertex_attr: {vertex_attr}")

g = Graph(n=args["nodes"], edges=edges, directed=True,
				vertex_attrs=vertex_attr)

assert(g.is_dag())

# drawing graphs
layout = g.layout("kamada_kawai")
plot(g, layout=layout)

print("[INFO] Starting dynamic model...")
system = DynamicManufacturing(g, args["seed"], rng =np.random.default_rng( args["seed"]))

# run the dynamic simulation and output the results to the defined medium
with open("event_log.txt", "w") as event_log:
	with open("log.txt", "w") as u:
		with sys.stdout if args["output"] == "-" else open(args["output"], "w") as f:
			production = 0
			runs = 0
			while production < 100:
				print("[INFO] minuts passed: {}, production: {}".format(runs, production))

				production = production + system.iterate(f, args["output"], event_log=event_log, log=u)[0]
				runs = runs + 1

print("[INFO] Generating event log...")

df = pd.read_csv('event_log.txt', sep=',',header=0)
df = df.drop(columns=['product_id'])
df = df.sort_values(by=['case_id', 'time_stamp'])
df['time_stamp_out'] = df['time_stamp'].shift(-1)
df = df[df.activity_id != "End of Line"]
df['product_id'] = df['case_id']
df.to_csv('event_log.txt', sep=',', index=False)

print("[INFO] Done!")
