from whyis.plugin import Plugin, NanopublicationListener
from whyis.database import driver
from whyis.namespace import NS
import weaviate
from weaviate.embedded import EmbeddedOptions
from flask import current_app
import rdflib
import json
import collections
from functools import reduce
import numpy as np
from weaviate.classes.query import Filter
import functools

whyis = NS.whyis

class VectorDBException(Exception):
    '''Raise an exception when something goes wrong in the vector database.'''

class VectorSpace:

    _collection = None

    field_names = ["subject", "graph", "extent", "v"]

    def __init__(self, vs_resource, db):
        self.resource = vs_resource
        self.identifier = vs_resource.identifier
        self.db = db
        self.collection_id = self.resource.value(NS.dc.identifier).value
        self.index_name = self.collection_id
        #self.field_names.append(self.vector_field_name)
        self.extent = self.resource.value(NS.whyis.hasExtent)
        self.extent = tuple(json.loads(str(self.extent)))
        self.dimensions = reduce(lambda a, b: a*b, self.extent)
        print(self.extent, self.dimensions)

        self.distance_metric = self.resource.value(whyis.hasDistanceMetric, Literal('cosine'))
        if self.distance_metric is not None:
            self.distance_metric = self.distance_metric.value
        self.index_type = self.resource.value(whyis.hasIndexType)
        if self.index_type is not None:
            self.index_type = str(self.index_type)

        self.field_names.extend(list(self.resource[whyis.hasDynamicField]))
        
    @property
    def collection(self):
        if self._collection is None:
            self.db.client.collections.create(
                self.collection_id,
                vector_index_config=Configure.VectorIndex.dynamic(
                    distance_metric=VectorDistances.COSINE
                ),
                properties = [
                    Property(
                        name="subject", data_type=DataType.TEXT, tokenization=Tokenization.FIELD
                    ),
                    Property(
                        name="graph", data_type=DataType.TEXT, tokenization=Tokenization.FIELD
                    ),
                ]
            )
        return self._collection

    def prepare_result(self, o):
        entity = dict(o.properties)
        entity['uuid'] = o.uuid
        if hasattr(o, 'vector'):
            entity['tensor'] = self.db.to_tensor(o.vector["default"], self.extent).tolist()
            
        return entity

    def _create_filter(self, subject=None, graph=None):
        filters = [Filter.by_property(k).equal(v) for k,v
                   in [('subject',subject), ('graph',graph)]
                   if kv[1] is not None]
        filters = functools.reduce(lambda a, b: a & b, filters)
        
    
    def similar(self, subject, graph=None, limit=10, offset=0):
        matches = self.get(subject, graph, include_vector=False)
        results = []
        for match in matches:
            response = self.collection.query.near_object(
                near_object = match['uuid']
                limit=limit,
                offset=offset,
                include_vector=True,
                return_metadata=MetadataQuery(distance=True),
            )
            results = []
            for o in response.objects:
                entity = self.prepare_result(o)
                entity['distance'] = o.metadata.distance
                entity['search_graph'] = graph
                entity['search_uuid'] = match.uuid
                entity['search_subject'] = subject
                results.append(entity)
            if len(matches) > 1:
                results = sorted(results, key=lambda x: x['distance'])
        return results

    def get(self, subject=None, graph=None, include_vector=True):
        response = self.collection.query.fetch_objects(self.create_filter(subject, graph),
                                                       include_vector=include_vector)
        results = [self.prepare_result(r) for r in response.objects]

        return results

    def search(self, tensor, limit, offset=0):
        vector, extent = self.db.to_vector(tensor)
        response = jeopardy.query.near_vector(
            near_vector=vector,
            limit=limit,
            offset=offset
            include_vector=True,
            return_metadata=MetadataQuery(distance=True)
        )
        print("searching %s"%self.identifier)
        i = 0
        results = []
        for o in response.objects:
            entity = self.prepare_result(o)
            entity['distance'] = o.metadata.distance
            entity['rank'] = i + offset
            i += 1
            results.append(entity)

        print("finished searching %s"%self.identifier)
        return results

@driver('weaviate')
class WeaviateDatabase(NanopublicationListener):

    formats = {
        'json' : NS.rdf.JSON
    }

    vector_parsers = {
        NS.mediaTypes['application/json'] : lambda d: d,
        NS.rdf.JSON : lambda d: d,
    }

    vector_serializers = {
        NS.mediaTypes['application/json'] :
            lambda a: a,
        NS.rdf.JSON :
            lambda a: a,
    }

    _spaces = None

    def __enter__(self):
        return self.client

    def __exit__(self, type, value, traceback):
        self.client.close()


    def __init__(self, config):
        self.db_name = config.get('_database', 'whyis')
        self.user = config.get('_username', None)
        self.password = config.get('_password', None)
        self.name = config.get('_name', "weaviate")
        self.host = config.get('_hostname',None)
        self.port = config.get('_port', 8080)
        self.grpc_port = config.get('_grpc_port', 50051)
        if host is None:
            self.port = 8079
            self.grpc_port = 50050
            self.embedded = True
        else:
            self.embedded = False

        if self.embedded:
            try:
                self.client = weaviate.connect_to_local(
                    host=self.host,
                    port=self.port,
                    grpc_port = self.grpc_port
                )
            except weaviate.exceptions.WeaviateConnectionError:
                self.client = weaviate.WeaviateClient(
                    embedded_options=EmbeddedOptions(
                        additional_env_vars={
                            ASYNC_INDEXING = 'true'
                        }
                    )
                )
                self.client.connect()
        else:
            # TODO for authentication see https://weaviate.io/developers/weaviate/connections/connect-local
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port = self.grpc_port
            )

    @property
    def spaces(self):
        if self._spaces is None:
            self._spaces = {}
            for uri in current_app.vocab.subjects(rdflib.RDF.type, NS.whyis.VectorSpace):
                print("Adding vector space",uri)
                resource = current_app.vocab.resource(uri)
                self._spaces[uri] = VectorSpace(resource, self)
        return self._spaces

    def on_publish(self, nanopub):
        g = rdflib.ConjunctiveGraph(store=nanopub.store)
        assertion_uri = str(nanopub.assertion.identifier)
        for space in self.spaces.values():
            inserts = []
            removes = []
            with space.collection.batch.dynamic() as batch:
                for s, p, o in nanopub.assertion.triples((None,space.identifier,None)):
                    data = self.parse_vector(o)
                    tensor = data['tensor']
                    vector, extent = self.to_vector(tensor)
                    if space.extent is not None and extent != space.extent:
                        raise VectorDBException(
                            "Tensor shape (%s) does not match schema tensor shape (%s)." % (extent, space.extent)
                        )
                    batch.add_object(
                        properties = dict(subject = str(s), graph = str(nanopub.assertion.identifier))
                        vector = vector
                    )

    def on_retire(self, nanopub):
        collection.data.delete_many(where=Filter.by_property("graph").equal(str(nanopub.assertion.identifier)))

    def search(self, space, vector, limit, offset=0):
        try:
            s = self.spaces[rdflib.URIRef(space)]
            return s.search(vector, limit, offset=offset)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

    def get(self, space=None, subject=None, graph=None):
        spaces = [space] if space is not None or len(space) == 0 else self.spaces.keys()
        try:
            spaces = [self.spaces[rdflib.URIRef(s)] for s in spaces]
            print(spaces)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

        results = []
        for s in spaces:
            results.extend(s.get(subject, graph))
        return results

    def similar(self, space=None, subject=None, graph=None, limit=10, offset=0):
        spaces = [space] if space is not None or len(space) == 0 else self.spaces.keys()
        try:
            spaces = [self.spaces[rdflib.URIRef(s)] for s in spaces]
            print(spaces)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

        results = []
        for s in spaces:
            results.extend(s.similar(subject, graph, limit, offset))
        return results


    def get_vectors(self, space=None, uri=None, graph=None):
        results = []
        for space in self.spaces.values():
            results.extend(space.get(uri, graph))
        return result

    def parse_vector(self, node):
        data = self.vector_parsers[node.datatype](node.value)
        print(type(data))
        if not isinstance(data, dict):
            data = dict(tensor=data)
        return data

    def serialize_vector(self, tensor, format):
        if format not in self.vector_serializers:
            format = self._formats[format]
        serializer = self.vector_serializers[format]
        return Literal(serializer(tensor), datatype=format)

    def to_vector(self, tensor):
        tensor = np.array(tensor)
        extent = tensor.shape
        vector = tensor.flatten()
        return vector, extent

    def to_tensor(self, vector, extent):
        return np.array(vector).reshape(extent)

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.disconnect(self.name)

class WeaviatePlugin(Plugin):

    def create_blueprint(self):
        return None

    def init(self):
        rdflib.term.bind(
            datatype=rdflib.RDF.JSON,
            pythontype=list,
            constructor=json.loads,
            lexicalizer=json.dumps,
            datatype_specific=True
        )
        rdflib.term.bind(
            datatype=NS.mediaTypes['application/json'],
            pythontype=list,
            constructor=json.loads,
            lexicalizer=json.dumps,
            datatype_specific=True
        )
        #driver(MilvusDatabase, "milvus")
