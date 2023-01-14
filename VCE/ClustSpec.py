from onsager import cluster


class ClusterSpecies(object):

    def __init__(self, specList, siteList, zero=True):
        """
        Creation to represent clusters from site Lists and species lists
        :param specList: Species lists
        :param siteList: Sites which the species occupy. Must be ClusterSite object from Onsager (D.R. Trinkle)
        :param zero: Whether to make the centroid of the sites at zero or not.
        """
        if len(specList)!= len(siteList):
            raise ValueError("Species and site lists must have same length")
        if not all(isinstance(site, cluster.ClusterSite) for site in siteList):
            raise TypeError("The sites must be entered as clusterSite object instances")
        if len(set(siteList)) != len(siteList):
            raise TypeError("Non-unique sites detected in sitelist: {}".format(siteList))

        # Calculate the translation to bring center of the sites to the origin unit cell if zero cluster indicated
        self.zero = zero
        if zero:
            Rtrans = sum([site.R for site in siteList])//len(siteList)
            self.transPairs = [(site-Rtrans, spec) for site, spec in zip(siteList, specList)]
        else:
            self.transPairs = [(site, spec) for site, spec in zip(siteList, specList)]
        # self.transPairs = sorted(self.transPairs, key=lambda x: x[1] + x[0].R[0] + x[0].R[1] + x[0].R[2])
        self.SiteSpecs = self.transPairs #sorted(self.transPairs, key=lambda s: np.linalg.norm(s[0].R))
        self.siteList = [site for site, spec in self.SiteSpecs]
        self.specList = [spec for site, spec in self.SiteSpecs]
        hashval = 0
        for site, spec in self.transPairs:
            hashval ^= hash((site, spec))
        self.__hashcache__ = hashval

    def __eq__(self, other):

        if set(self.SiteSpecs) == set(other.SiteSpecs):
            return True
        return False

    def __hash__(self):
        return self.__hashcache__

    def g(self, crys, gop):
        specList = [spec for site, spec in self.SiteSpecs]
        siteList = [site.g(crys, gop) for site, spec in self.SiteSpecs]
        return self.__class__(specList, siteList, zero=self.zero)

    @staticmethod
    def inSuperCell(SpCl, N_units):
        siteList = []
        specList = []
        for site, spec in SpCl.SiteSpecs:
            Rnew = site.R % N_units
            siteNew = cluster.ClusterSite(ci=site.ci, R=Rnew)
            siteList.append(siteNew)
            specList.append(spec)
        return SpCl.__class__(specList, siteList, zero=SpCl.zero)


    def strRep(self):
        str= ""
        for site, spec in self.SiteSpecs:
            str += "Spec:{}, site:{},{} ".format(spec, site.ci, site.R)
        return str

    def __repr__(self):
        return self.strRep()

    def __str__(self):
        return self.strRep()