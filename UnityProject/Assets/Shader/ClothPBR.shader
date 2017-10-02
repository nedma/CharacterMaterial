Shader "WeNZ/ClothPBR" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)

	_FuzzColor ("FuzzColor", Color) = (1,1,1,1)
	_Cloth ("Cloth", Range(0.01,1)) = 0.8
	_LobeEnergy1 ("_LobeEnergy1", Range(0.01,5)) = 0.8
	_LobeEnergy2 ("_LobeEnergy2", Range(0.01,5)) = 0.8

	_MainTex ("Base (RGB)", 2D) = "white" {}
	//_MaskTex ("Light mask (RG) Matelness(B)", 2D) = "white" {}
	_MetalnessTex ("Metalness", 2D) = "white" {}
	_RoughnessTex ("Roughness", 2D) = "white" {}
	_BumpMap ("Normal (Normal)", 2D) = "bump" {}

	_Roughness ("Roughness Value", Range(0.01,1)) = 0.028
	_Fresnel ("Fresnel Value", Range(0.01,1)) = 0.028

	_SpeuclarEnvMap("Reflection Map", Cube) = "_Skybox" { TexGen CubeReflect }
	_DiffuseEnvColor("DiffuseEnvColor", Color) = (1,1,1,1)
	_EnvLightIntensity ("EnvLightIntensity", Range(0,5)) = 0.5


	_WorldSpaceLightDir("WorldSpaceLightDir", Vector) = (1,3,1,0)
	_LightColor("LightColor", Color) = (0.5, 0.5, 0.5, 1)
}


SubShader
    {
        Tags { "Queue" = "Geometry" "IgnoreProjector" = "True" }
        LOD 400
		//CULL Off

        Pass
        {
        	Name "FORWARD"
			Tags { "LightMode" = "ForwardBase" }
        
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
	    #pragma target 3.0
            #pragma multi_compile_fwdbase nolightmap

            #define UNITY_PASS_FORWARDBASE
            #include "UnityCG.cginc"
			#include "Lighting.cginc"
			#include "AutoLight.cginc"
			#include "BRDF_Unreal.cginc"

            #define USE_REAL_LIGHT
			#define GAMMA_CORRECTION
            
            // vertex-to-fragment interpolation data
			struct v2f_surf {
			  float4 pos : SV_POSITION;
			  float2 pack0 : TEXCOORD0;
			  fixed3 lightDir : TEXCOORD1;
			  fixed3 vlight : TEXCOORD2;
			  float3 viewDir : TEXCOORD3;
			  LIGHTING_COORDS(5,6)
			  
		  		half3 TtoW0 : TEXCOORD7;
				half3 TtoW1 : TEXCOORD8;
				half3 TtoW2 : TEXCOORD9;
			};
			
			struct Input
            {
                float2 uv_MainTex;
                half3 TtoW0;
				half3 TtoW1;
				half3 TtoW2;
				float3 worldRefl;
            };
            
            fixed4 _Color;
            sampler2D _MainTex, _BumpMap;//, _Beckmann;
            //sampler2D _MaskTex;
	    sampler2D _MetalnessTex;
	    sampler2D _RoughnessTex;

			sampler2D unity_Lightmap;
			sampler2D unity_LightmapInd;
			
            float4 _MainTex_ST;
			samplerCUBE _SpeuclarEnvMap;
			fixed3 _DiffuseEnvColor;
            float _Fresnel,_Roughness,_GaussConstant, _EnvLightIntensity;
			fixed4 _WorldSpaceLightDir;
			fixed3 _LightColor;
            

			fixed4 _FuzzColor;
			fixed _Cloth;
			fixed _LobeEnergy1;
			fixed _LobeEnergy2;

			struct SurfaceOutputCustom
			{
				half3 Albedo;
				half3 Normal;
				half3 Emission;			
				half Specular;
				half Gloss;
				half Alpha;
				half Metalness;
				half Roughness;
			};

             void surf (Input IN, inout SurfaceOutputCustom o)
            {
                fixed4 albedo = tex2D(_MainTex, IN.uv_MainTex);
#ifdef GAMMA_CORRECTION
				albedo.rgb = albedo.rgb * albedo.rgb;
#endif
                o.Albedo = albedo.rgb * _Color;
                o.Alpha = albedo.a;

				//fixed4 reflcol = texCUBE (_SpeuclarEnvMap, IN.worldRefl);
				//o.Emission = reflcol.rgb;

				//fixed4 mask = tex2D(_MaskTex, IN.uv_MainTex);
				//o.Emission += _IllumColorR * mask.r *_IllumIntensityR + _IllumColorG * mask.g *_IllumIntensityG;

                o.Normal = UnpackNormal(tex2D(_BumpMap, IN.uv_MainTex));            
                o.Normal = half3(dot(IN.TtoW0, o.Normal.xyz), dot(IN.TtoW1, o.Normal.xyz), dot(IN.TtoW2, o.Normal.xyz));             
                o.Normal = normalize(o.Normal);

				o.Metalness = tex2D(_MetalnessTex, IN.uv_MainTex) * _Fresnel;
				o.Roughness = tex2D(_RoughnessTex, IN.uv_MainTex) * _Roughness;
            }

			float3 ClothShading( SurfaceOutputCustom s, float3 L, float3 V, float atten )
			{
				float3 H = normalize(V + L);
				float NoL = saturate( dot(s.Normal, L) );
				float NoV = saturate( abs( dot(s.Normal, V) ) + 1e-5 );
				float NoH = saturate( dot(s.Normal, H) );
				float VoH = saturate( dot(V, H) );

				// Diffuse	
				float3 Diffuse = Diffuse_Lambert( s.Albedo );
				float3 Diff = Diffuse * _LobeEnergy2;

				// Cloth - Asperity Scattering - Inverse Beckmann Layer	
				float3 F1 = F_Schlick( _FuzzColor, VoH );
				float  D1 = D_InvGGX( s.Roughness, NoH );
				float  V1 = Vis_Cloth( NoV, NoL );

				float3 Spec1 = D1 * V1 * F1;

				// Generalized microfacet specular
				float3 F2 = F_Schlick( _Color, VoH );
				float  D2 = D_GGX( s.Roughness, NoH ) * _LobeEnergy1;
				float  V2 = Vis_SmithJointApprox( s.Roughness, NoV, NoL );

				float3 Spec2 = D2 * V2 * F2;

				float3 Spec = lerp(Spec2, Spec1, _Cloth);

				return (Diff + Spec) * NoL * _LightColor0.rgb * atten;
			}
 
            inline fixed4 LightingCookTorrance (SurfaceOutputCustom s, fixed3 lightDir, fixed3 viewDir, fixed atten)
            {
                //clip ( s.Alpha - _Cutoff );
 
                viewDir = normalize ( viewDir );
                lightDir = normalize ( lightDir );
                float3 h = normalize ( lightDir + viewDir );
                
                //float3 r = normalize(reflect(-viewDir, s.Normal));
                
                float NdotL_unsat = dot ( s.Normal, lightDir );
                float NdotH_unsat = dot ( s.Normal, h );
                float NdotL = saturate( NdotL_unsat );
                float NdotH = saturate( NdotH_unsat );
                float NdotV = saturate( dot ( s.Normal, viewDir ) );
                float VdotH = saturate( dot ( viewDir, h ) );
                
                
 
                float geo_numerator = 2.0 * NdotH;
                float geo_b = ( geo_numerator * NdotV ) / VdotH;
                float geo_c = ( geo_numerator * NdotL ) / VdotH;
                float geo = min( 1.0f, min( geo_b, geo_c ) );
 

				// Vis_Cloth
				float visCloth = 1.0 / ( 4 * ( NdotL + NdotV - NdotL * NdotV ) );
 				
 
 				////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //float roughness = tex2D( _Beckmann, float2 ( NdotH_unsat * 0.5 + 0.5, s.Roughness ) ).r;
                
               
                
                //////////////
                // beckmann distribution function
				float roughness = s.Roughness;
                float _RoughnessSqr = roughness * roughness;
                float NdotHSqr = NdotH * NdotH;
		        float r1 = 1.0 / ( 3.14 * _RoughnessSqr * NdotHSqr * NdotHSqr + 0.01f);
		        float r2 = (NdotHSqr - 1.0) / (_RoughnessSqr * NdotHSqr + 0.01f);
		        float roughness_beckmann = r1 * exp(r2);
				//roughness_beckmann = saturate(roughness_beckmann);
                
				// InvGGX
				float a = roughness * roughness;
				float a2 = a * a;
				float A = 4;
				float d = ( NdotH - a2 * NdotH ) * NdotH + a2;
				float item1 = 1 / (3.1415 * (1 + A*a2));
				float cos2 = NdotH * NdotH;
				float sin2 = 1 - cos2;
				float cot2 = cos2 / sin2;
				float item2 = (A * exp(-1.0 * cot2 / a2)) / (sin2 * sin2); 
				//float item2 = A * a2*a2 / d*d;
				float invGGX = item1 * (1 + item2);
                
                //////////////
                // Blinn's NDF
                //float alpha = acos(NdotH);
				//float roughness_Blinn = _GaussConstant*exp(-(alpha*alpha)/(roughness * roughness));
                
                
                //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
                float fresnel = pow( 1.0 - VdotH, 5.0 );

				float matFresnel = s.Metalness; // RF (0◦)
                fresnel *= ( 1.0 - matFresnel );
                fresnel += matFresnel;
 
 				float3 Cspec = lerp(float3(0.3,0.3,0.3), s.Albedo, matFresnel);
                float3 spec = float3 ( (fresnel * geo * roughness_beckmann ) / ( NdotV * NdotL + 0.02)); // 分母的pi一般被认为是包含到light参数里面去了所以这里不算
				//spec  *= Cspec;
				

 				float3 diff = (1 - fresnel) * s.Albedo;
 
				//
				//float3 refDirW =  normalize( reflect(viewDir, s.Normal));
				//float3 speuclarEnvRadiance = texCUBE(_SpeuclarEnvMap, refDirW);

				float4 refDirW;
				refDirW.xyz = normalize( reflect(viewDir, s.Normal));
				refDirW.w = roughness_beckmann * 8;;
				float3 speuclarEnvRadiance = texCUBEbias(_SpeuclarEnvMap, refDirW);
				speuclarEnvRadiance = speuclarEnvRadiance * _EnvLightIntensity;

                fixed4 c;

#ifdef USE_REAL_LIGHT
				fixed3 lightColor = _LightColor0.rgb;
#else
				fixed3 lightColor = _LightColor.rgb;
#endif


                //c.rgb = NdotL * (lightColor.rgb * atten + speuclarEnvRadiance * _EnvLightIntensity)  * ( spec +  diff);
                
                
                // AO
                //fixed3 aoForDiff = ShadeSH9 (float4(s.Normal, 1.0));
                //fixed3 aoForSpec = ShadeSH9 (float4(r, 1.0));
                
				// Direct lighting
				float3 lightSpec = NdotL * (lightColor.rgb * atten)  * spec * Cspec;
				float3 lightDiffuse = NdotL * (lightColor.rgb * atten)  * diff;

				// Indirect lighting
				float3 envSpec =  NdotL * speuclarEnvRadiance * Cspec * spec;
               	//float3 envDiff = NdotL * diffuseEnvRadiance  * diff;
				float3 envDiff = _DiffuseEnvColor * diff; // use a const color to represent the env diffuse light
				
				//float3 clothSpec = float3(visCloth * invGGX) * float3(fresnel * Cspec);
				float3 clothSpec = float3(invGGX * fresnel) * Cspec;

				c.rgb = 0.0f;
				//c.rgb += lightSpec;
				c.rgb += clothSpec * (NdotL * _LightColor0.rgb * atten);
				//c.rgb += lightDiffuse;
				//c.rgb += envSpec;
				//c.rgb += envDiff;

				//c.rgb += s.Emission;
                
                c.a = s.Alpha;
                 
                
                //return float4(aoForSpec,1);
                //return float4(aoForDiff,1);     
				//return  float4(speuclarEnvRadiance * _EnvLightIntensity, 1);          
                //return roughness;
				//return float4(s.Normal, 1);

				//return float4(invGGX,invGGX,invGGX, 1);
				return c;
            }
            
           v2f_surf vert (appdata_full v) 
           {
       			v2f_surf o;

				o.pos = mul (UNITY_MATRIX_MVP, v.vertex);
				o.pack0.xy = TRANSFORM_TEX(v.texcoord, _MainTex);

				float3 worldN = mul((float3x3)_Object2World, SCALED_NORMAL);

				o.viewDir = WorldSpaceViewDir(v.vertex);
				
				float3 binormal = cross(v.normal, v.tangent.xyz) * v.tangent.w;
				float3x3 rotation = float3x3(v.tangent.xyz, binormal, v.normal);
				o.TtoW0 = mul(rotation, ((float3x3)_Object2World)[0].xyz)*unity_Scale.w;
				o.TtoW1 = mul(rotation, ((float3x3)_Object2World)[1].xyz)*unity_Scale.w;
				o.TtoW2 = mul(rotation, ((float3x3)_Object2World)[2].xyz)*unity_Scale.w;
	
#ifdef USE_REAL_LIGHT
				o.lightDir = WorldSpaceLightDir(v.vertex);
#else
				o.lightDir = _WorldSpaceLightDir.xyz;
#endif


				// SH/ambient and vertex lights

				float3 shlight = ShadeSH9 (float4(worldN,1.0));
				o.vlight = shlight;
				#ifdef VERTEXLIGHT_ON
				float3 worldPos = mul(_Object2World, v.vertex).xyz;
				o.vlight += Shade4PointLights (
				unity_4LightPosX0, unity_4LightPosY0, unity_4LightPosZ0,
				unity_LightColor[0].rgb, unity_LightColor[1].rgb, unity_LightColor[2].rgb, unity_LightColor[3].rgb,
				unity_4LightAtten0, worldPos, worldN );
				#endif // VERTEXLIGHT_ON


				// pass lighting information to pixel shader
				TRANSFER_VERTEX_TO_FRAGMENT(o);
				return o;
			}
            
			fixed4 frag (v2f_surf IN) : SV_Target 
			{
				Input surfIN = (Input)0;
				surfIN.uv_MainTex = IN.pack0.xy;
				surfIN.TtoW0 = IN.TtoW0;
				surfIN.TtoW1 = IN.TtoW1;
				surfIN.TtoW2 = IN.TtoW2;
				
				SurfaceOutputCustom o = (SurfaceOutputCustom)0;
				o.Albedo = 0.0;
				o.Emission = 0.0;
				o.Specular = 0.0;
				o.Alpha = 0.0;
				o.Gloss = 0.0;
				o.Metalness = 0.0;
				o.Roughness = 0.0;

				// call surface function
				surf (surfIN, o);

				// compute lighting & shadowing factor
#ifdef USE_REAL_LIGHT
				fixed atten = LIGHT_ATTENUATION(IN);
#else
				fixed atten = 1;
#endif
				
				fixed4 c = 0;

				// realtime lighting: call lighting function
				c = LightingCookTorrance (o, IN.lightDir, normalize(half3(IN.viewDir)), atten);
				
				
				// Unity input SH
				c.rgb += o.Albedo * IN.vlight;


			
				c.a = o.Alpha;

#ifdef GAMMA_CORRECTION
				c.rgb = sqrt(c.rgb);
#endif
				return c;
			}

            ENDCG
        }
    }


FallBack "WeNZ/VehiclePrimitive"
}
