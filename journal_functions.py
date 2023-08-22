

def create_new_file_nx(nx_part_file_name: str):
    theSession = NXOpen.Session.GetSession()

    listing_window = theSession.ListingWindow

    fileNew1 = theSession.Parts.FileNew()
    fileNew1.TemplateFileName = "model-plain-1-mm-template.prt"
    fileNew1.UseBlankTemplate = False
    fileNew1.ApplicationName = "ModelTemplate"
    fileNew1.Units = NXOpen.Part.Units.Millimeters
    fileNew1.RelationType = ""
    fileNew1.UsesMasterModel = "No"
    fileNew1.TemplateType = NXOpen.FileNewTemplateType.Item
    fileNew1.TemplatePresentationName = "Model"
    fileNew1.ItemType = ""
    fileNew1.Specialization = ""
    fileNew1.SetCanCreateAltrep(False)
    fileNew1.NewFileName = nx_part_file_name
    listing_window.Open()
    listing_window.WriteFullline(f'{fileNew1.NewFileName = }')
    listing_window.Close()
    fileNew1.MasterFileName = ""
    fileNew1.MakeDisplayedPart = True
    fileNew1.DisplayPartOption = NXOpen.DisplayPartOption.AllowAdditional
    nXObject1 = fileNew1.Commit()
    fileNew1.Destroy()
    return theSession


def create_bezier_curve_from_ctrlpts(ctrlpt_dict: dict, theSession):
    workPart = theSession.Parts.Work

    listing_window = theSession.ListingWindow

    te_lines = []
    te_points = []
    upper_splines = []
    lower_splines = []

    for ctrlpts in ctrlpt_dict.values():  # ctrlpts is a 3-D array and ctrlpt_dict represents all the airfoils
        for idx, coord_set in enumerate(ctrlpts):  # coord_set is a 2-D array representing the curve
            studioSplineBuilderEx1 = workPart.Features.CreateStudioSplineBuilderEx(NXOpen.NXObject.Null)
            studioSplineBuilderEx1.MatchKnotsType = NXOpen.Features.StudioSplineBuilderEx.MatchKnotsTypes.General
            studioSplineBuilderEx1.Type = NXOpen.Features.StudioSplineBuilderEx.Types.ByPoles
            studioSplineBuilderEx1.IsSingleSegment = True
            for pt in coord_set:
                scalar1 = workPart.Scalars.CreateScalar(pt[0], NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                scalar2 = workPart.Scalars.CreateScalar(pt[1], NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                scalar3 = workPart.Scalars.CreateScalar(pt[2], NXOpen.Scalar.DimensionalityType.NotSet,
                                                        NXOpen.SmartObject.UpdateOption.WithinModeling)
                point = workPart.Points.CreatePoint(scalar1, scalar2, scalar3,
                                                    NXOpen.SmartObject.UpdateOption.WithinModeling)
                point.RemoveViewDependency()
                geometricConstraintData = studioSplineBuilderEx1.ConstraintManager.CreateGeometricConstraintData()
                geometricConstraintData.Point = point
                studioSplineBuilderEx1.ConstraintManager.Append(geometricConstraintData)

            nXObject = studioSplineBuilderEx1.Commit()
            if idx == 0:
                upper_splines.append(nXObject)
            else:
                lower_splines.append(nXObject)
            studioSplineBuilderEx1.Destroy()
        te_1 = NXOpen.Point3d(ctrlpts[0][0][0], ctrlpts[0][0][1], ctrlpts[0][0][2])
        te_2 = NXOpen.Point3d(ctrlpts[-1][-1][0], ctrlpts[-1][-1][1], ctrlpts[-1][-1][2])
        te_points.append([te_1, te_2])
        te_lines.append(workPart.Curves.CreateLine(te_1, te_2))

    return len(ctrlpt_dict.keys()), te_lines, te_points, upper_splines, lower_splines, listing_window, workPart


def through_curve_builder(n_airfoil_sections, te_lines, te_points, upper_splines, lower_splines, listing_window,
                          workPart):
    listing_window.Open()
    listing_window.WriteFullline('Creating loft through curves...')
    listing_window.Close()

    time1 = time.time()

    throughCurvesBuilder1 = workPart.Features.CreateThroughCurvesBuilder(NXOpen.Features.Feature.Null)
    throughCurvesBuilder1.Alignment.AlignCurve.DistanceTolerance = 4e-6
    throughCurvesBuilder1.Alignment.AlignCurve.ChainingTolerance = 4e-6
    throughCurvesBuilder1.SectionTemplateString.DistanceTolerance = 4e-6
    throughCurvesBuilder1.SectionTemplateString.ChainingTolerance = 4e-6

    sections = [NXOpen.Section.Null] * n_airfoil_sections

    for idx in range(n_airfoil_sections):
        section1 = workPart.Sections.CreateSection(4e-6, 4e-6, 0.5)
        throughCurvesBuilder1.SectionsList.Append(section1)
        section1.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.CurvesAndPoints)
        selectionIntentRuleOptions1 = workPart.ScRuleFactory.CreateRuleOptions()
        selectionIntentRuleOptions1.SetSelectedFromInactive(False)

        curves1 = [NXOpen.IBaseCurve.Null] * 1
        line1 = te_lines[idx]
        curves1[0] = line1
        curveDumbRule1 = workPart.ScRuleFactory.CreateRuleBaseCurveDumb(curves1, selectionIntentRuleOptions1)

        selectionIntentRuleOptions1.Dispose()
        section1.AllowSelfIntersection(False)
        section1.AllowDegenerateCurves(False)

        rules1 = [None] * 1
        rules1[0] = curveDumbRule1
        helpPoint1 = te_points[idx][0]
        section1.AddToSection(rules1, line1, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint1,
                              NXOpen.Section.Mode.Create, False)

        for idx2 in range(2):

            selectionIntentRuleOptions2 = workPart.ScRuleFactory.CreateRuleOptions()
            selectionIntentRuleOptions2.SetSelectedFromInactive(False)

            features1 = [NXOpen.Features.Feature.Null] * 1

            if idx2 == 0:
                splines = upper_splines
            else:
                splines = lower_splines

            studioSpline1 = splines[idx]
            features1[0] = studioSpline1
            curveFeatureRule1 = workPart.ScRuleFactory.CreateRuleCurveFeature(features1, NXOpen.DisplayableObject.Null,
                                                                              selectionIntentRuleOptions2)

            selectionIntentRuleOptions2.Dispose()
            section1.AllowSelfIntersection(False)
            section1.AllowDegenerateCurves(False)

            rules2 = [None] * 1
            rules2[0] = curveFeatureRule1
            spline1 = studioSpline1.FindObject("CURVE 1")
            helpPoint2 = te_points[idx][1]
            section1.AddToSection(rules2, spline1, NXOpen.NXObject.Null, NXOpen.NXObject.Null, helpPoint2,
                                  NXOpen.Section.Mode.Create, False)

        sections[idx] = section1

    listing_window.Open()
    listing_window.WriteFullline(f'{len(sections) = }')
    listing_window.Close()

    throughCurvesBuilder1.Alignment.SetSections(sections)

    throughCurvesBuilder1.CommitFeature()
    throughCurvesBuilder1.Destroy()

    time2 = time.time()

    listing_window.Open()
    listing_window.WriteFullline(f'Created loft through curves ({time2 - time1:.4f} seconds).')
    listing_window.Close()
    return workPart


def save_file(workPart):
    partSaveStatus1 = workPart.Save(NXOpen.BasePart.SaveComponents.TrueValue, NXOpen.BasePart.CloseAfterSave.FalseValue)
    partSaveStatus1.Dispose()


