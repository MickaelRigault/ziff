""" """

import dask
from dask import distributed


def _not_delayed_(func):
    """ Doing nothing """
    return func
    
def get_delayed_func(use_dask=True):
    """ """
    return dask.delayed if use_dask else _not_delayed_

class DaskCluster( object ):
    """ """
    def __init__(self, client=None):
        """ """
        if client is not None:
            self.set_client(client)

    def set_client(self, client):
        """ """
        self._client = client

    # --------- #
    #  SET      #
    # --------- #
    def setup_cluster(self, which="ccin2p3", scale=None, **kwargs):
        """ Set the client from the cluster information """
        cluster = self.get_cluster( which="ccin2p3", scale=None, **kwargs)
        self.set_client( distributed.Client(cluster) )
            
    # --------- #
    #  GETTER   #
    # --------- #
    @staticmethod
    def get_cluster(which="ccin2p3", scale=None, set_client=True, **kwargs):
        """ """
        
        if which == "ccin2p3":
            from dask_jobqueue import SGECluster
            prop = dict(name="dask-worker",  walltime="06:00:00",
                        memory='8GB', death_timeout=120, project="P_ztf",
                        resource_spec='sps=1', cores=1, processes=1)
            
            cluster = SGECluster(**{**prop,**kwargs})
        else:
            raise NotImplementedError(f"only 'ccin2p3' cluster implemented {which} given")

        if scale is not None:
            cluster.scale( int(scale) )

        return cluster

    def get_inactive_workers(self):
        """ """
        if not self.has_client():
            raise AttributeError("No client set.")
        
        return [k for k,v in self.client.processing().values() if len(v)==0]

    def get_client(self, client=None):
        """ """
        if client is None and not self.has_client():
            warnings.warn("No dask client given and no dask client sent to the instance.")
            return None

        return client if client is None else self.client
    
    # ================= #
    #    Properties     #
    # ================= #
    @property
    def client(self):
        """ """
        if not self.has_client():
            return None
        return self._client

    def has_client(self):
        """ """
        return hasattr(self, "_client") and self._client is not None
