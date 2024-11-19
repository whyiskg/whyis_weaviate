from whyis.plugin import Plugin, NanopublicationListener
from whyis.database import driver
from whyis.namespace import NS
#import weaviate
#from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Tokenization
#from weaviate.classes.query import MetadataQuery
#from weaviate.embedded import EmbeddedOptions
from flask import current_app
import rdflib
import json
import collections
from functools import reduce
import numpy as np
#from weaviate.classes.query import Filter
import functools
import os
import requests

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
        #print(self.extent, self.dimensions)

        self.distance_metric = self.resource.value(whyis.hasDistanceMetric)
        if self.distance_metric is not None:
            self.distance_metric = self.distance_metric.value
        else:
            self.distance_metric = "cosine"
        self.index_type = self.resource.value(whyis.hasIndexType)
        if self.index_type is not None:
            self.index_type = str(self.index_type)

        self.field_names.extend(list(self.resource[whyis.hasDynamicField]))
        
    @property
    def collection(self):
        if self._collection is None:
            self._collection = requests.get(f'{self.db.endpoint}schema/{self.collection_id}')
            if self._collection.status_code == 404:
                requests.post(f'{self.db.endpoint}schema/', json={
                    'class' : self.collection_id,
                    'vectorIndexType' : 'hnsw',
                    'vectorIndexConfig' : dict(
                        distanceMetric=self.distance_metric
                    ),
                    'properties' : [
                        dict(
                            name="subject", dataType='text', tokenization='field'
                        ),
                        dict(
                            name="graph", dataType='text', tokenization='field'
                        ),
                    ]
                })
                self._collection = requests.get(f'{self.db.endpoint}schema/{self.collection_id}')
                print(self._collection)
        return self._collection

    def prepare_result(self, o, include_vector=True):
        entity = dict(o)
        additional = entity['_additional']
        del entity['_additional']
        entity.update(additional)
        if include_vector:
            entity['tensor'] = self.db.to_tensor(additional['vector'], self.extent).tolist()
            del entity['vector']
            
        return entity

    def _create_filter(self, subject=None, graph=None):
        filters = [f'''{{
  path: ["{k}"],
  operator: Equal,
  valueText: "{v}"
}}
''' for k,v in [('subject',subject), ('graph',graph)]
                   if v is not None]
        return ', '.join(filters)
    
    def similar(self, subject, graph=None, limit=10, offset=0):
        matches = self.get(subject, graph, include_vector=False)
        results = []
        for match in matches:
            query = f'''{{
  Get {{
    {self.collection_id}(
      nearObject: {{
        id: "{match['id']}"
      }},
      limit: {limit},
      offset: {offset}
    ) {{
      subject
      graph
      _additional {{
        id
        vector
        distance
      }}
    }}
  }}
}}'''
            response = requests.post(f'{self.db.endpoint}graphql', json={'query' : query})
            if response.status_code != 200:
                raise VectorDBException(response.text)
            data = response.json()
            if 'errors' in data:
                raise VectorDBException(response.text)
            responses = data['data']['Get'][self.collection_id]
            for r in responses:
                entity = self.prepare_result(r)
                entity['search_graph'] = graph
                entity['search_uuid'] = match['id']
                entity['search_subject'] = subject
                entity['space'] = str(self.identifier)
                results.append(entity)
        if len(matches) > 1:
            results = sorted(results, key=lambda x: x['distance'])
        return results

    def get(self, subject=None, graph=None, include_vector=True):
        f = self._create_filter(subject, graph)
        query = f'''{{
  Get {{
    {self.collection_id}(where: {{
      operator: And,
      operands: [{f}]
    }}) {{
      subject
      graph
      _additional {{
        id
        vector
      }}
    }}
  }}
}}'''
        response = requests.post(f'{self.db.endpoint}graphql', json={'query' : query})
        if response.status_code != 200:
            raise VectorDBException(response.text)
        data = response.json()
        if 'errors' in data:
            raise VectorDBException(response.text)

        responses = data['data']['Get'][self.collection_id]
        results = [self.prepare_result(r, include_vector) for r in responses]

        return results

    def search(self, tensor, limit, offset=0):
        vector, extent = self.db.to_vector(tensor)
        json_vector = json.dumps(vector.tolist())
        query = f'''{{
  Get {{
    {self.collection_id}(
      nearVector: {{
       vector: {json_vector}
      }},
      limit: {limit},
      offset: {offset}
    ) {{
      subject
      graph
      _additional {{
        id
        vector
        distance
      }}
    }}
  }}
}}'''
        response = requests.post(f'{self.db.endpoint}graphql', json={'query' : query})
        if response.status_code != 200:
            raise VectorDBException(response.text)
        data = response.json()
        if 'errors' in data:
            raise VectorDBException(response.text)
        responses = data['data']['Get'][self.collection_id]
        i = 0
        results = []
        for r in responses:
            entity = self.prepare_result(r)
            entity['space'] = str(self.identifier)
            entity['rank'] = i + offset
            i += 1
            results.append(entity)
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

#    def __enter__(self):
#        return self.client

#    def __exit__(self, type, value, traceback):
#        self.client.close()


    def __init__(self, config):
        self.db_name = config.get('_database', 'whyis')
        self.user = config.get('_username', None)
        self.password = config.get('_password', None)
        self.name = config.get('_name', "weaviate")
        self.host = config.get('_hostname','localhost')
        self.port = config.get('_port', 8080)

        self.endpoint = f'http://{self.host}:{self.port}/v1/'
        print(self.endpoint)
        
        r = requests.get(self.endpoint)
        if r.status_code != 200:
            raise VectorDBExcption(r.text)

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
        payload = []
        for space in self.spaces.values():
            inserts = []
            removes = []
            for s, p, o in nanopub.assertion.triples((None,space.identifier,None)):
                data = self.parse_vector(o)
                tensor = data['tensor']
                vector, extent = self.to_vector(tensor)
                if space.extent is not None and extent != space.extent:
                    raise VectorDBException(
                        "Tensor shape (%s) does not match schema tensor shape (%s)." % (extent, space.extent)
                    )
                payload.append({
                    'class' : space.collection_id,
                    'properties' : dict(
                        subject = str(s),
                        graph = str(nanopub.assertion.identifier)
                    ),
                    'vector' : vector.tolist()
                })
        if len(payload) > 0:
            response = requests.post(f'{self.endpoint}batch/objects',
                                     json={
                                         "fields": ["ALL"],
                                         "objects":payload
                                     })
            if response.status_code != 200:
                raise VectorDBException(response.text)

    def on_retire(self, nanopub):
        for space in self.spaces:
            if (None, space.identifier, None) in nanopub.assertion:
                response = requests.delete(
                    f'{self.endpoint}batch/objects',
                    json={
                        'match' : {
                            'class' : space.collection_id,
                            'where' : {
                                'operator' : 'Equal',
                                'path' : ['graph'],
                                'valueText' : str(nanopub.assertion.identifier)
                            }
                        }
                    }
                )
                if response.status_code != 200:
                    raise VectorDBException(response.text)

    def search(self, space, vector, limit, offset=0):
        try:
            s = self.spaces[rdflib.URIRef(space)]
            return s.search(vector, limit, offset=offset)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

    def get(self, space=None, subject=None, graph=None):
        spaces = [space] if space is not None and len(space) == 0 else self.spaces.keys()
        try:
            spaces = [self.spaces[rdflib.URIRef(s)] for s in spaces]
            #print(spaces)
        except KeyError:
            raise VectorDBException("Vector Space does not exist")

        results = []
        for s in spaces:
            results.extend(s.get(subject, graph))
        return results

    def similar(self, space=None, subject=None, graph=None, limit=10, offset=0):
        if not space or len(space) == 0:
            spaces =  self.spaces.keys()
        else:
            spaces = [space]
        try:
            spaces = [self.spaces[rdflib.URIRef(s)] for s in spaces]
            #print(spaces)
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
        #print(type(data))
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

#    def __exit__(self, exc_type, exc_value, traceback):
#        self.connection.disconnect(self.name)

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
