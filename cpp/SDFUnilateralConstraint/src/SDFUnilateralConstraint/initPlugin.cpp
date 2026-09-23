#include <SDFUnilateralConstraint/config.h>

#include <sofa/core/ObjectFactory.h>

#include <string>

using sofa::core::ObjectFactory;

extern "C"
{
SDFUNILATERALCONSTRAINT_API void initExternalModule();
SDFUNILATERALCONSTRAINT_API const char* getModuleName();
SDFUNILATERALCONSTRAINT_API const char* getModuleVersion();
SDFUNILATERALCONSTRAINT_API const char* getModuleLicense();
SDFUNILATERALCONSTRAINT_API const char* getModuleDescription();
SDFUNILATERALCONSTRAINT_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
        first = false;
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(SDFUNILATERALCONSTRAINT_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "MCR SDF unilateral constraint prototype for SOFA 21.12.";
}

const char* getModuleComponentList()
{
    static std::string classes =
        ObjectFactory::getInstance()->listClassesFromTarget(
            sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}
