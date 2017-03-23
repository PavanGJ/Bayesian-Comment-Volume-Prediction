import sys
sys.path.append('../')
from Structure import Structure

class StructureTest:
    def __init__(self):
        self.network = Structure()
        self.network.add_edge(["Intelligence","Grade"],types=["d","lgandd"])
        self.network.add_edge(["Difficulty","Grade"],types=["lg","lgandd"])
        self.network.add_edge(["Intelligence","SAT"],types=["d","lgandd"])
        self.network.add_edge(["Grade","Letter"],types=["lgandd","lg"])

    def test_get_structure(self):
        print(self.network.get_structure())

    def test_get_parents(self):
        print("Parents of Grade:",self.network.get_parents("Grade"))
        print("Parents of SAT:",self.network.get_parents("SAT"))
        print("Parents of Letter:",self.network.get_parents("Letter"))

    def test_get_children(self):
        print("Children of Intelligence:",self.network.get_children("Intelligence"))
        print("Children of Difficulty:",self.network.get_children("Difficulty"))
        print("Children of Grade:",self.network.get_children("Grade"))
        print("Children of Letter:",self.network.get_children("Letter"))

    def test_get_vertex_type(self):
        print("Type of vertex-Intelligence:",self.network.get_vertex_type("Intelligence"))
        print("Type of vertex-Difficulty:",self.network.get_vertex_type("Difficulty"))
        print("Type of vertex-Grade:",self.network.get_vertex_type("Grade"))
        print("Type of vertex-SAT:",self.network.get_vertex_type("SAT"))
        print("Type of vertex-Letter:",self.network.get_vertex_type("Letter"))

if __name__ == "__main__":
    test = StructureTest()
    test.test_get_parents()
    test.test_get_children()
    test.test_get_vertex_type()
