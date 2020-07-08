
SELECT  
	FORMAT(A.DateKey,'MM') AS MTH,
	YEAR(A.DateKey) AS YR, C.RegionCountryName, C.ContinentName, D.ProductName , 
	SUM(A.UnitCost) AS UNITCOST, SUM(A.UnitPrice) AS UNITPRICE, SUM(A.SalesQuantity) AS SALESQUANTITY, 
	SUM(A.TotalCost) AS TOTALCOST, SUM(A.SalesAmount) AS SALESAMOUNT,
	CONCAT(CONVERT(varchar(10),YEAR(A.DateKey)),CONVERT(varchar(10),FORMAT(A.DateKey,'MM'))) AS YRMONTH

	FROM ContosoRetailDW.dbo.FactSales A

	LEFT JOIN ContosoRetailDW.dbo.DimStore B ON A.StoreKey= B.StoreKey
	LEFT JOIN ContosoRetailDW.dbo.DimGeography C ON B.GeographyKey = C.GeographyKey
	LEFT JOIN ContosoRetailDW.dbo.DimProduct D ON A.ProductKey = D.ProductKey

	WHERE C.RegionCountryName IN ('India', 'China', 'Portgal', 'United Kingdom', 'Australia', 'Canada', 'Germany')
	GROUP BY A.DateKey, C.RegionCountryName, C.ContinentName, D.ProductName
	
	ORDER BY YRMONTH, C.ContinentName, C.RegionCountryName;




